"""入库：解析 → 切分 → 向量化 → 写库。"""

from pathlib import Path

from sqlalchemy.engine import Engine

from .chunking import chunk_markdown
from .config import Settings
from .embeddings import get_embedding_client
from .parsing import iter_resume_files, parse_file
from .store import init_db, insert_document, reset_db


def ingest_folder(engine: Engine, settings: Settings, folder: Path, reset: bool = False) -> dict:
    """入库一个目录下的全部支持文件；同名 source 已存在则跳过。"""
    if reset:
        reset_db(engine)

    embedder = get_embedding_client(settings)
    init_db(engine, embedder.dim)

    files = iter_resume_files(folder)
    docs = 0
    chunks_total = 0
    for path in files:
        text = parse_file(path)
        pieces = chunk_markdown(text)
        if not pieces:
            continue
        vectors = embedder.embed_batch([p.content for p in pieces])
        rows = [
            (p.content, p.section, p.index, vec)
            for p, vec in zip(pieces, vectors, strict=True)
        ]
        title = _title_of(text, path)
        insert_document(engine, str(path), title, rows)
        docs += 1
        chunks_total += len(rows)
    return {"docs": docs, "chunks": chunks_total}


def _title_of(text: str, path: Path) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip() or path.stem
    return path.stem
