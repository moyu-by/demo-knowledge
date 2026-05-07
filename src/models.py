from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Note:
    id: str
    title: str
    content: str
    file_path: str
    tags: List[str] = field(default_factory=list)
    category: str = ""
    keywords: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    chunk_index: int = 0

    @property
    def is_chunk(self) -> bool:
        return self.chunk_index > 0

    def brief(self, max_len: int = 100) -> str:
        return f"[{self.category}] {self.title} — {self.content[:max_len].strip()}..."
