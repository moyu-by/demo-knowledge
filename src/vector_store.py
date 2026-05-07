"""ChromaDB 向量存储封装. 使用 sentence-transformers 本地嵌入."""
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.models import Note


class _LocalEmbedFn:
    """sentence-transformers 本地嵌入函数，适配 ChromaDB。"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self._model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    @staticmethod
    def name() -> str:
        return "sentence-transformers"

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input, show_progress_bar=False).tolist()

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input, show_progress_bar=False).tolist()

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input, show_progress_bar=False).tolist()


class VectorStore:
    """向量存储封装，支持增删查。"""

    def __init__(self, storage_dir: str, model_name: Optional[str] = None, device: str = "cpu"):
        self.client = chromadb.PersistentClient(
            path=storage_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        emb_fn = _LocalEmbedFn(model_name or "sentence-transformers/all-MiniLM-L6-v2", device)
        self.collection = self.client.get_or_create_collection(
            name="notes",
            embedding_function=emb_fn,
        )

    # ---- write ----

    def add_notes(self, notes: List[Note]) -> None:
        if not notes:
            return
        # 过滤已存在的 ID
        existing = set(self.collection.get()["ids"])
        to_add = [n for n in notes if n.id not in existing]
        if not to_add:
            return

        ids = [n.id for n in to_add]
        documents = [n.content for n in to_add]
        metadatas: List[Dict[str, str]] = [
            {
                "title": n.title,
                "file_path": n.file_path,
                "tags": ",".join(n.tags),
                "category": n.category or "",
                "keywords": ",".join(n.keywords),
                "chunk_index": str(n.chunk_index),
                "created_at": n.created_at or "",
                "updated_at": n.updated_at or "",
            }
            for n in to_add
        ]
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def update_metadata(self, note: Note) -> None:
        """更新单条笔记的元数据（如处理后的 tags/category 等）。"""
        self.collection.update(
            ids=[note.id],
            metadatas=[{
                "title": note.title,
                "file_path": note.file_path,
                "tags": ",".join(note.tags),
                "category": note.category or "",
                "keywords": ",".join(note.keywords),
                "chunk_index": str(note.chunk_index),
                "created_at": note.created_at or "",
                "updated_at": note.updated_at or "",
            }],
        )

    def delete_by_file(self, file_path: str) -> None:
        hits = self.collection.get(where={"file_path": file_path})
        if hits["ids"]:
            self.collection.delete(ids=hits["ids"])

    def clear(self) -> None:
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)

    # ---- read ----

    def search(self, query: str, top_k: int = 3, where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        results = self.collection.query(query_texts=[query], n_results=top_k, where=where)
        if not results["ids"] or not results["ids"][0]:
            return []
        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - (results["distances"][0][i] if results["distances"] else 0),
            })
        return items

    def get_all_tags(self) -> Dict[str, int]:
        """统计所有标签及其出现次数。"""
        metas = self.collection.get()["metadatas"]
        counter: Dict[str, int] = {}
        for m in metas:
            for t in m.get("tags", "").split(","):
                if t:
                    counter[t] = counter.get(t, 0) + 1
        return counter

    def get_all_categories(self) -> Dict[str, int]:
        metas = self.collection.get()["metadatas"]
        counter: Dict[str, int] = {}
        for m in metas:
            cat = m.get("category", "")
            if cat:
                counter[cat] = counter.get(cat, 0) + 1
        return counter

    def stats(self) -> Dict[str, int]:
        info = self.collection.get()
        return {
            "chunks": len(info["ids"]),
            "notes": len({m.get("file_path", "") for m in info["metadatas"]}),
        }
