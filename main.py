#!/usr/bin/env python3
"""个人知识库 / 笔记整理 Agent — CLI 入口。

用法:
  python main.py import <目录>    导入笔记目录
  python main.py process [--batch N]  自动打标签/分类
  python main.py query "<问题>"     自然语言问答
  python main.py report [--days 7]   周报/总结
  python main.py topic "<主题>"      专题知识总结
  python main.py tags             查看所有标签
  python main.py list             列出笔记概览
  python main.py stats            知识库统计
  python main.py delete [--all | --file <路径>]  删除笔记索引
"""
import argparse
import os
import sys
from pathlib import Path

# 确保项目根在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.loader import load_notes, chunk_all
from src.vector_store import VectorStore
from src.processor import NoteProcessor
from src.retriever import Retriever
from src.generator import Generator


def _api_base(cfg):
    """解析 API 地址，优先 config，其次环境变量 ANTHROPIC_BASE_URL。"""
    return (cfg.get("llm", "api_base") or os.environ.get("ANTHROPIC_BASE_URL") or "")


def cmd_import(args, cfg):
    notes_dir = args.directory
    vs = VectorStore(
        storage_dir=cfg.get("kb", "storage_dir") or "./kb_storage",
        model_name=cfg.get("embedding", "model"),
        device=cfg.get("embedding", "device") or "cpu",
    )
    notes = load_notes(notes_dir)
    if not notes:
        print(f"[!] 目录 '{notes_dir}' 中没有找到 .md 文件")
        return

    print(f"📄 发现 {len(notes)} 篇笔记，正在切块并导入向量库...")
    chunks = chunk_all(
        notes,
        max_tokens=cfg.get("chunking", "max_tokens") or 500,
        overlap=cfg.get("chunking", "overlap") or 50,
    )
    vs.add_notes(chunks)
    st = vs.stats()
    print(f"✅ 导入完成：{st['notes']} 篇笔记，{st['chunks']} 个片段")


def cmd_process(args, cfg):
    vs = VectorStore(
        storage_dir=cfg.get("kb", "storage_dir") or "./kb_storage",
        model_name=cfg.get("embedding", "model"),
        device=cfg.get("embedding", "device") or "cpu",
    )

    # 从已导入的向量库中获取笔记列表（重新读取源文件）
    notes_dir = cfg.get("kb", "notes_dir") or "./data/notes"
    all_notes = load_notes(notes_dir)
    if not all_notes:
        print("[!] 没有找到笔记，请先运行 import")
        return

    processor = NoteProcessor(
        api_key=cfg.get("llm", "api_key"),
        model=args.model or cfg.get("llm", "model") or "deepseek-v4-flash",
        base_url=_api_base(cfg),
    )

    # 只处理未处理的笔记
    todo = processor.unprocessed_notes(all_notes, vs)

    if not todo:
        print("✨ 所有笔记都已处理过标签和分类")
        return

    # 切块后取第一个 chunk 做处理（避免重复处理同一笔记的多个 chunk）
    unique_notes = {}
    for n in todo:
        if n.file_path not in unique_notes:
            unique_notes[n.file_path] = n
    todo = list(unique_notes.values())

    batch = args.batch or 10
    print(f"🔖 正在处理 {len(todo)} 篇未分类笔记（批次大小 {batch}）...")
    ok, fail = processor.process(todo, vs, batch_size=batch)

    mode = "LLM" if not processor._local_mode else "本地规则"
    print(f"✅ 处理完成（模式: {mode}）: 成功 {ok}, 失败 {fail}")


def cmd_query(args, cfg):
    vs = VectorStore(
        storage_dir=cfg.get("kb", "storage_dir") or "./kb_storage",
        model_name=cfg.get("embedding", "model"),
        device=cfg.get("embedding", "device") or "cpu",
    )
    retriever = Retriever(
        vs,
        top_k=cfg.get("retrieval", "top_k") or 3,
    )
    gen = Generator(
        retriever,
        api_key=cfg.get("llm", "api_key"),
        model=args.model or cfg.get("llm", "model") or "deepseek-v4-flash",
        base_url=_api_base(cfg),
    )
    print(f"🔍 问题: {args.question}\n")
    answer = gen.answer(args.question, stream=True)
    if not gen._has_llm:
        print(answer)


def cmd_report(args, cfg):
    vs = VectorStore(
        storage_dir=cfg.get("kb", "storage_dir") or "./kb_storage",
        model_name=cfg.get("embedding", "model"),
        device=cfg.get("embedding", "device") or "cpu",
    )
    retriever = Retriever(vs, top_k=10)
    gen = Generator(
        retriever,
        api_key=cfg.get("llm", "api_key"),
        model=args.model or cfg.get("llm", "model") or "deepseek-v4-flash",
        base_url=_api_base(cfg),
    )
    print(f"📊 正在生成 {args.days} 天知识周报...\n")
    report = gen.weekly_report(days=args.days)
    print(report)


def cmd_topic(args, cfg):
    vs = VectorStore(
        storage_dir=cfg.get("kb", "storage_dir") or "./kb_storage",
        model_name=cfg.get("embedding", "model"),
        device=cfg.get("embedding", "device") or "cpu",
    )
    retriever = Retriever(vs, top_k=10)
    gen = Generator(
        retriever,
        api_key=cfg.get("llm", "api_key"),
        model=args.model or cfg.get("llm", "model") or "deepseek-v4-flash",
        base_url=_api_base(cfg),
    )
    print(f"📚 正在整理关于「{args.topic}」的知识总结...\n")
    summary = gen.topic_summary(args.topic)
    print(summary)


def cmd_tags(args, cfg):
    vs = VectorStore(
        storage_dir=cfg.get("kb", "storage_dir") or "./kb_storage",
        model_name=cfg.get("embedding", "model"),
        device=cfg.get("embedding", "device") or "cpu",
    )
    tags = vs.get_all_tags()
    cats = vs.get_all_categories()

    print("🏷️  标签统计:")
    for t, c in sorted(tags.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print(f"\n📂 分类统计:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")


def cmd_list(args, cfg):
    vs = VectorStore(
        storage_dir=cfg.get("kb", "storage_dir") or "./kb_storage",
        model_name=cfg.get("embedding", "model"),
        device=cfg.get("embedding", "device") or "cpu",
    )
    info = vs.collection.get()
    seen: dict = {}
    for m in info["metadatas"]:
        fp = m.get("file_path", "")
        if fp not in seen:
            seen[fp] = m
    print(f"📚 知识库共有 {len(seen)} 篇笔记:\n")
    for fp, m in sorted(seen.items(), key=lambda x: x[1].get("title", "")):
        title = m.get("title", "?")
        cat = m.get("category", "")
        tags = m.get("tags", "")
        tag_str = f" [{tags}]" if tags else ""
        cat_str = f" ({cat})" if cat else ""
        print(f"  • {title}{cat_str}{tag_str}")
        print(f"    {fp}")


def cmd_stats(args, cfg):
    vs = VectorStore(
        storage_dir=cfg.get("kb", "storage_dir") or "./kb_storage",
        model_name=cfg.get("embedding", "model"),
        device=cfg.get("embedding", "device") or "cpu",
    )
    st = vs.stats()
    print(f"📊 知识库统计")
    print(f"  {'笔记数':>8}: {st['notes']}")
    print(f"  {'片段数':>8}: {st['chunks']}")
    print(f"  {'存储位置':>8}: {cfg.get('kb', 'storage_dir')}")


def cmd_delete(args, cfg):
    vs = VectorStore(
        storage_dir=cfg.get("kb", "storage_dir") or "./kb_storage",
        model_name=cfg.get("embedding", "model"),
        device=cfg.get("embedding", "device") or "cpu",
    )
    if args.all:
        confirm = input("⚠️  确定要清空整个知识库？(y/n): ")
        if confirm.lower() == "y":
            vs.clear()
            print("✅ 已清空知识库")
    elif args.file:
        vs.delete_by_file(args.file)
        print(f"✅ 已删除: {args.file}")
    else:
        info = vs.collection.get()
        seen = {}
        for m in info["metadatas"]:
            fp = m.get("file_path", "")
            if fp and fp not in seen:
                seen[fp] = m
        items = sorted(seen.items(), key=lambda x: x[1].get("title", ""))
        if not items:
            print("（知识库为空）")
            return
        print("📚 选择要删除的笔记：")
        for i, (fp, m) in enumerate(items, 1):
            print(f"  {i}. {m.get('title', '?')}")
        print(f"  a. 全部删除")
        print(f"  0. 取消")
        choice = input(f"\n请输入编号 (1-{len(items)}): ").strip()
        if choice == "a":
            confirm = input("⚠️  确定清空整个知识库？(y/n): ")
            if confirm.lower() == "y":
                vs.clear()
                print("✅ 已清空知识库")
        elif choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(items):
                fp = items[idx - 1][0]
                vs.delete_by_file(fp)


def main():
    parser = argparse.ArgumentParser(
        description="个人知识库 / 笔记整理 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py import ./my-notes
  python main.py process
  python main.py query "什么是CAP定理？"
  python main.py report --days 7
  python main.py topic "机器学习"
  python main.py tags
  python main.py stats
        """,
    )
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    sub = parser.add_subparsers(dest="command")

    # import
    p_import = sub.add_parser("import", help="导入笔记目录")
    p_import.add_argument("directory", help="Markdown 笔记目录路径")

    # process
    p_proc = sub.add_parser("process", help="自动打标签/分类")
    p_proc.add_argument("--batch", type=int, default=10, help="每批处理笔记数")
    p_proc.add_argument("--model", help="LLM 模型（覆盖配置项）")

    # query
    p_query = sub.add_parser("query", help="自然语言问答")
    p_query.add_argument("question", help="你的问题")
    p_query.add_argument("--model", help="LLM 模型（覆盖配置项）")

    # report
    p_report = sub.add_parser("report", help="生成周报/总结")
    p_report.add_argument("--days", type=int, default=7, help="统计最近 N 天")
    p_report.add_argument("--model", help="LLM 模型（覆盖配置项）")

    # topic
    p_topic = sub.add_parser("topic", help="专题知识总结")
    p_topic.add_argument("topic", help="主题/标签名称")
    p_topic.add_argument("--model", help="LLM 模型（覆盖配置项）")

    # tags
    sub.add_parser("tags", help="查看所有标签和分类")

    # list
    sub.add_parser("list", help="列出笔记概览")

    # stats
    sub.add_parser("stats", help="知识库统计")

    # delete
    p_del = sub.add_parser("delete", help="删除笔记索引")
    p_del.add_argument("--file", help="按文件路径删除")
    p_del.add_argument("--all", action="store_true", help="清空整个知识库")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cfg = Config(args.config)

    cmds = {
        "import": cmd_import,
        "process": cmd_process,
        "query": cmd_query,
        "report": cmd_report,
        "topic": cmd_topic,
        "tags": cmd_tags,
        "list": cmd_list,
        "stats": cmd_stats,
        "delete": cmd_delete,
    }
    cmds[args.command](args, cfg)


if __name__ == "__main__":
    main()
