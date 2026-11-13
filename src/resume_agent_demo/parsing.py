"""简历文件解析：md / txt / pdf（pypdf）。"""

from pathlib import Path

_SUFFIXES = {".md", ".markdown", ".txt"}


def parse_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return _parse_pdf(path)
    raise ValueError(f"unsupported file type: {path.name}（支持 md/txt/pdf）")


def _parse_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def iter_resume_files(folder: Path) -> list[Path]:
    """递归收集支持的简历文件，按路径排序保证可重复。"""
    files: list[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in _SUFFIXES | {".pdf"}:
            files.append(p)
    return sorted(files, key=lambda p: str(p))
