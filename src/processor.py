"""LLM 自动打标签、分类、提取关键词。

支持两种模式：
1. LLM 模式 — 调用 Anthropic API（需设置 ANTHROPIC_API_KEY）
2. 本地模式 — 基于规则的关键词提取（零依赖，无需 API Key）
"""
import json
import os
import re
from typing import List, Optional, Tuple

from src.models import Note
from src.vector_store import VectorStore


def _extract_text(response):
    """从 Anthropic Message 响应中提取文本，跳过 ThinkingBlock。"""
    for block in response.content:
        if hasattr(block, 'text'):
            return block.text
    return ""

# 预定义分类关键词（本地模式用）
_CATEGORY_KEYWORDS: dict = {
    "技术": ["python", "javascript", "代码", "算法", "前端", "后端", "api", "docker",
              "数据库", "linux", "git", "框架", "函数", "类", "部署", "测试"],
    "学习": ["笔记", "教程", "课程", "读书", "阅读", "学习", "知识", "概念",
              "理论", "方法", "总结", "心得"],
    "工作": ["项目", "会议", "需求", "方案", "计划", "报告", "周报", "日报",
              "进度", "协作", "团队", "任务"],
    "生活": ["日常", "随笔", "旅行", "美食", "运动", "健康", "阅读", "电影",
              "音乐", "日记"],
    "随笔": ["感想", "思考", "想法", "随想", "杂记", "观点"],
}


class NoteProcessor:
    """自动处理笔记：打标签、分类、提取关键词。"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514",
                 base_url: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self._local_mode = not self.api_key

    # ==================== 公开入口 ====================

    def process(self, notes: List[Note], vs: VectorStore, batch_size: int = 10) -> Tuple[int, int]:
        """处理笔记列表，更新 vector store 元数据。返回 (成功数, 失败数)。"""
        ok = 0
        fail = 0
        for i in range(0, len(notes), batch_size):
            batch = notes[i : i + batch_size]
            if self._local_mode:
                results = [self._local_process(n) for n in batch]
            else:
                results = self._llm_process_batch(batch)
            for note, (tags, cat, keywords) in zip(batch, results):
                if tags or cat or keywords:  # 有结果就算成功
                    note.tags = tags or note.tags
                    note.category = cat or note.category
                    note.keywords = keywords or note.keywords
                    vs.update_metadata(note)
                    ok += 1
                else:
                    fail += 1
        return ok, fail

    def unprocessed_notes(self, notes: List[Note], vs: VectorStore) -> List[Note]:
        """筛选出尚未处理（没分类/标签）的笔记。"""
        existing = vs.stats().get("chunks", 0)
        if existing == 0:
            return notes
        # 收集已有分类的 file_path
        processed = set()
        all_metas = vs.collection.get()["metadatas"]
        for m in all_metas:
            if m.get("category", ""):
                processed.add(m.get("file_path", ""))
        return [n for n in notes if n.file_path not in processed]

    # ==================== 本地模式（规则） ====================

    def _local_process(self, note: Note):
        title_lower = note.title.lower()
        content_lower = note.content.lower()[:500]
        text = title_lower + " " + content_lower

        # 分类
        cat_scores = {}
        for cat, kws in _CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in kws if kw.lower() in text)
            if score > 0:
                cat_scores[cat] = score
        category = max(cat_scores, key=cat_scores.get) if cat_scores else "其他"

        # 关键词：TF 统计
        words = re.findall(r"[\w一-鿿]+", text)
        from collections import Counter
        stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
                       "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
                       "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
                       "它", "们", "那", "里", "为", "与", "及", "等", "之", "能",
                       "the", "a", "an", "is", "are", "was", "were", "to", "in",
                       "it", "of", "for", "on", "and", "or", "not", "be", "with"}
        filtered = [w for w in words if len(w) > 1 and w not in stop_words]
        top_kws = [w for w, _ in Counter(filtered).most_common(5)]

        # 标签：从分类和关键词中提取
        tags = [category] if category != "其他" else []
        tags.extend([kw for kw in top_kws[:3] if kw not in tags])

        return tags, category, top_kws[:5]

    # ==================== LLM 模式（Anthropic） ====================

    def _llm_process_batch(self, notes: List[Note]):
        import anthropic

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = anthropic.Anthropic(**kwargs)
        notes_block = "\n\n".join(
            f"[笔记{i+1}] {n.title}\n{n.content[:400]}"
            for i, n in enumerate(notes)
        )
        prompt = f"""你是一个笔记整理助手。请为以下笔记提取信息，仅返回 JSON 数组。

对每篇笔记提取：
- tags：2-4 个标签（中文或英文）
- category：一个分类（技术/学习/工作/生活/随笔/项目/其他）
- keywords：3-5 个关键词（中文或英文）

返回格式（严格 JSON 数组）：
[
  {{"index":1,"tags":["tag1","tag2"],"category":"技术","keywords":["kw1","kw2"]}},
  ...
]

笔记：
{notes_block}"""

        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            text = _extract_text(resp)
            return self._parse_llm_response(text, len(notes))
        except Exception as e:
            print(f"[WARN] LLM 调用失败: {e}，回退到本地模式")
            return [self._local_process(n) for n in notes]

    def _parse_llm_response(self, text: str, expected: int):
        json_str = text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return [([], "", [])] * expected
        results = []
        for item in data:
            results.append((
                item.get("tags", []),
                item.get("category", ""),
                item.get("keywords", []),
            ))
        # 补齐缺失
        while len(results) < expected:
            results.append(([], "", []))
        return results[:expected]
