"""检索器：语义搜索 + 按笔记聚合。"""
from typing import Dict, List, Optional

from src.vector_store import VectorStore


class Retriever:
    """封装搜索逻辑，支持按文件聚合结果。"""

    def __init__(self, vector_store: VectorStore, top_k: int = 3):
        self.vs = vector_store
        self.top_k = top_k

    def search(self, query: str, top_k: Optional[int] = None, where: Optional[Dict] = None) -> List[Dict]:
        """语义搜索，返回按 score 降序的匹配片段。"""
        k = top_k or self.top_k
        return self.vs.search(query, top_k=k, where=where)

    def search_grouped(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """搜索并按笔记原文聚合（同一文件合并）。"""
        results = self.search(query, top_k=top_k or self.top_k * 2)
        groups: Dict[str, Dict] = {}
        for r in results:
            fp = r["metadata"].get("file_path", "")
            if fp in groups:
                groups[fp]["content"] += "\n\n" + r["content"]
                groups[fp]["score"] = max(groups[fp]["score"], r["score"])
            else:
                groups[fp] = {
                    "file_path": fp,
                    "title": r["metadata"].get("title", ""),
                    "content": r["content"],
                    "tags": r["metadata"].get("tags", ""),
                    "category": r["metadata"].get("category", ""),
                    "score": r["score"],
                }
        return sorted(groups.values(), key=lambda x: x["score"], reverse=True)[:self.top_k]
