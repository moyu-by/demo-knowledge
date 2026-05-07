"""生成器：问答、周报/学习总结。"""
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.retriever import Retriever
from src.vector_store import VectorStore


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _days_ago(n: int) -> str:
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


def _extract_text(response):
    """从 Anthropic Message 响应中提取文本，跳过 ThinkingBlock。"""
    for block in response.content:
        if hasattr(block, 'text'):
            return block.text
    return ""


class Generator:
    """基于检索结果的 LLM 生成（问答 / 周报 / 专题总结）。"""

    def __init__(self, retriever: Retriever, api_key: Optional[str] = None,
                 model: str = "claude-sonnet-4-20250514", base_url: Optional[str] = None):
        self.retriever = retriever
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self._has_llm = bool(self.api_key)

    # ==================== 问答 ====================

    def answer(self, question: str, stream: bool = False) -> str:
        """基于知识库回答自然语言问题。"""
        context_items = self.retriever.search(question)
        if not context_items:
            return "（知识库中没有找到相关内容。）"

        context = "\n\n---\n\n".join(
            f"[来源: {c['metadata'].get('title','')}]\n{c['content']}"
            for c in context_items
        )

        if not self._has_llm:
            return self._format_no_llm(question, context, context_items)

        return self._llm_answer(question, context, stream=stream)

    def _make_client(self):
        import anthropic
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return anthropic.Anthropic(**kwargs)

    def _llm_answer(self, question: str, context: str, stream: bool = False) -> str:
        client = self._make_client()
        prompt = f"""你是一个个人知识库问答助手。基于以下笔记内容回答问题。

要求：
1. 如果笔记内容不足以回答，如实说不知道
2. 引用具体笔记内容作为依据
3. 用中文回答，保持简洁

笔记内容：
{context}

问题：{question}"""
        if stream:
            return self._llm_answer_stream(client, prompt)
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_text(resp)
        except Exception as e:
            return f"（LLM 调用失败: {e}）\n\n相关笔记片段：\n{context}"

    def _llm_answer_stream(self, client, prompt: str) -> str:
        full = ""
        try:
            with client.messages.stream(
                model=self.model,
                max_tokens=1500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for chunk in stream.text_stream:
                    print(chunk, end="", flush=True)
                    full += chunk
            print()
            return full
        except Exception as e:
            return f"\n（LLM 调用失败: {e}）"

    def _format_no_llm(self, question: str, context: str, items: List[Dict]) -> str:
        lines = [f"问题: {question}", "", "相关笔记:", ""]
        for c in items:
            title = c["metadata"].get("title", "未命名")
            lines.append(f"  📄 {title}")
            lines.append(f"     {c['content'][:200]}...")
            lines.append("")
        return "\n".join(lines)

    # ==================== 周报 ====================

    def weekly_report(self, days: int = 7) -> str:
        """按分类汇总近期笔记，生成周报。"""
        since = _days_ago(days)

        # 优先按分类组织
        cats = self.retriever.vs.get_all_categories()
        if not cats:
            # 无分类时，直接列出所有笔记
            info = self.retriever.vs.collection.get()
            if not info["ids"]:
                return "（知识库中暂无笔记。）"
            titles = list({m.get("title", ""): m for m in info["metadatas"]}.keys())
            lines = [f"# 知识库周报（{since} ~ {_today()}）\n"]
            lines.append(f"共 {len(titles)} 篇笔记：\n")
            for t in titles:
                lines.append(f"- {t}")
            return "\n".join(lines)

        report_parts: List[str] = []
        for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
            if count == 0:
                continue
            results = self.retriever.search("", top_k=count, where={"category": cat})
            # 收集该分类下的笔记标题
            titles = list({r["metadata"].get("title", "") for r in results if r["metadata"].get("category") == cat})
            # 收集内容片段
            contents = [r["content"][:300] for r in results[:3]]

            block = f"## {cat}（{len(titles)} 篇）\n"
            for t in titles:
                block += f"- {t}\n"
            if contents:
                block += "\n内容摘要：\n" + "\n".join(f"> {c}" for c in contents)
            report_parts.append(block)

        full = f"# 知识库周报（{since} ~ {_today()}）\n\n"
        full += "\n\n".join(report_parts)

        if self._has_llm:
            return self._llm_summarize(full)
        return full

    def _llm_summarize(self, report: str) -> str:
        client = self._make_client()
        prompt = f"""请基于以下笔记周报草稿，生成一份简洁的周报总结。

格式：
1. 本周概述（1-2句话）
2. 各分类要点（每个分类 1-2 句）
3. 建议关注方向

原始数据：
{report}"""
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_text(resp)
        except Exception:
            return report

    # ==================== 专题总结 ====================

    def topic_summary(self, topic: str) -> str:
        """对某个主题/标签进行专题总结。"""
        # 先按标签搜
        results = self.retriever.search(topic, top_k=10)
        if not results:
            return f"（没有找到关于「{topic}」的笔记。）"

        context = "\n\n---\n\n".join(
            f"[{c['metadata'].get('title','')}] {c['content'][:500]}"
            for c in results
        )

        if not self._has_llm:
            notes_list = "\n".join(
                f"- {c['metadata'].get('title','')} (相关度: {c['score']:.2f})"
                for c in results
            )
            return f"## 关于「{topic}」的笔记\n\n{notes_list}"

        client = self._make_client()
        prompt = f"""请基于以下笔记内容，整理一份关于「{topic}」的知识总结。

要求：
1. 归纳核心要点
2. 列出关键概念
3. 指出笔记间的关联
4. 如有不同观点，一并呈现

笔记内容：
{context}"""
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_text(resp)
        except Exception as e:
            return f"（LLM 调用失败: {e}）\n\n相关笔记：\n{context[:1000]}"
