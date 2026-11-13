"""Markdown/纯文本切分：按标题分节，节内按段落打包，超长段落硬切并带重叠。"""

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    content: str
    section: str
    index: int


def chunk_markdown(text: str, max_chars: int = 600, overlap: int = 80) -> list[Chunk]:
    if not text or not text.strip():
        return []

    chunks: list[Chunk] = []
    for title, body in _split_sections(text):
        for block in _pack(body, max_chars, overlap):
            content = f"{title}\n\n{block}" if title else block
            chunks.append(Chunk(content=content, section=title, index=len(chunks)))
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    title = ""
    body_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        if body:
            sections.append((title, body))

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            body_lines = []
            title = m.group(2).strip()
        else:
            body_lines.append(line)
    flush()
    return sections


def _pack(body: str, max_chars: int, overlap: int) -> list[str]:
    """按段落贪心打包；超长段落按窗口硬切（步长 max-overlap）。"""
    units: list[tuple[str, bool]] = []  # (text, is_continuation)
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append((para, False))
        else:
            step = max(1, max_chars - overlap)
            for i in range(0, len(para), step):
                piece = para[i : i + max_chars]
                units.append((piece, i > 0))
                if i + max_chars >= len(para):
                    break

    blocks: list[str] = []
    current = ""
    for piece, cont in units:
        if not current:
            current = piece
        elif len(current) + 2 + len(piece) <= max_chars:
            current = f"{current}\n\n{piece}"
        else:
            tail = current[-overlap:] if 0 < overlap < len(current) else current
            blocks.append(current)
            current = (tail + piece) if cont else (tail + "\n" + piece)
    if current:
        blocks.append(current)
    return blocks
