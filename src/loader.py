"""加载本地 Markdown 笔记文件."""
import re
import hashlib
from pathlib import Path
from typing import List, Optional

from src.models import Note


def load_notes(notes_dir: str) -> List[Note]:
    """递归扫描目录下所有 .md 文件并解析为 Note 对象。"""
    notes: List[Note] = []
    base = Path(notes_dir).expanduser().resolve()
    if not base.exists():
        print(f"[WARN] 笔记目录不存在: {base}")
        return notes

    for fp in sorted(base.rglob("*.md")):
        note = _parse_markdown(fp)
        if note:
            notes.append(note)
    return notes


def _parse_markdown(filepath: Path) -> Optional[Note]:
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return None
    if not content.strip():
        return None

    title = filepath.stem
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    tags: List[str] = []

    # --- 解析 YAML frontmatter ---
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if fm_match:
        fm_text = fm_match.group(1)
        for line in fm_text.split("\n"):
            line = line.strip()
            if line.startswith("title:"):
                title = line.split(":", 1)[1].strip().strip("\"'")
            elif line.startswith("date:") or line.startswith("created:"):
                created_at = line.split(":", 1)[1].strip()
            elif line.startswith("updated:"):
                updated_at = line.split(":", 1)[1].strip()
            elif line.startswith("tags:"):
                raw = line.split(":", 1)[1].strip()
                if raw.startswith("["):
                    tags = [t.strip().strip("\"'") for t in raw.strip("[]").split(",") if t.strip()]
                else:
                    tags = [t.strip() for t in raw.split(",") if t.strip()]
            elif line.startswith("category:"):
                pass  # 提前解析的分类可后续利用
        content = content[fm_match.end() :]

    note_id = hashlib.md5(str(filepath).encode()).hexdigest()[:12]
    return Note(
        id=note_id,
        title=title,
        content=content.strip(),
        file_path=str(filepath),
        tags=tags,
        created_at=created_at,
        updated_at=updated_at or created_at,
    )


def chunk_note(note: Note, max_tokens: int = 500, overlap: int = 50) -> List[Note]:
    """将长笔记按 token 数切块，返回多个 Note（chunk）。"""
    words = note.content.split()
    if len(words) <= max_tokens:
        return [note]

    # 粗略估算：1 token ≈ 1.3 中文词
    step = max(int((max_tokens - overlap) / 1.3), 1)
    limit = int(max_tokens / 1.3)

    chunks: List[Note] = []
    for i in range(0, len(words), step):
        seg = words[i : i + limit]
        if len(seg) < limit // 4:
            continue
        chunk = Note(
            id=f"{note.id}_c{(i // step)}",
            title=note.title,
            content=" ".join(seg),
            file_path=note.file_path,
            tags=note.tags.copy(),
            category=note.category,
            keywords=note.keywords.copy(),
            created_at=note.created_at,
            updated_at=note.updated_at,
            chunk_index=i // step,
        )
        chunks.append(chunk)
    return chunks


def chunk_all(notes: List[Note], max_tokens: int = 500, overlap: int = 50) -> List[Note]:
    result: List[Note] = []
    for n in notes:
        result.extend(chunk_note(n, max_tokens, overlap))
    return result
