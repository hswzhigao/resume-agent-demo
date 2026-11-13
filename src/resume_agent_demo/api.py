"""FastAPI：/ingest /search /match /health。"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .graph import run_match
from .ingest import ingest_folder
from .retriever import search
from .store import make_engine

_INDEX_HTML = Path(__file__).parent / "static" / "index.html"


class IngestRequest(BaseModel):
    folder: str | None = None  # 默认 data/sample
    reset: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class MatchRequest(BaseModel):
    jd: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


def create_app(settings: Settings | None = None, engine=None) -> FastAPI:
    settings = settings or get_settings()
    engine = engine if engine is not None else make_engine(settings.database_url)

    app = FastAPI(title="resume-agent-demo", version="0.1.0", root_path=settings.root_path)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        """操作页面（单文件 HTML，fetch 相对路径以兼容反代前缀）。"""
        return _INDEX_HTML.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "embedding": "provider" if settings.has_embedding else "local-hash",
            "llm": "on" if settings.has_llm else "heuristic",
            "rerank": "on" if settings.has_rerank else "off",
            "mask_pii": settings.mask_pii,
        }

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict[str, Any]:
        default_dir = Path(__file__).resolve().parents[2] / "data" / "sample"
        folder = Path(req.folder) if req.folder else default_dir
        if not folder.exists():
            raise HTTPException(status_code=400, detail=f"folder not found: {folder}")
        return ingest_folder(engine, settings, folder, reset=req.reset)

    @app.post("/search")
    def search_resumes(req: SearchRequest) -> dict[str, Any]:
        hits = search(engine, settings, req.query, top_k=req.top_k)
        return {
            "query": req.query,
            "results": [
                {
                    "title": h.doc_title,
                    "source": h.source,
                    "section": h.section,
                    "content": h.content,
                    "score": round(h.score, 4),
                    "vector_rank": h.vector_rank,
                    "keyword_rank": h.keyword_rank,
                }
                for h in hits
            ],
        }

    @app.post("/match")
    def match(req: MatchRequest) -> dict[str, Any]:
        try:
            return run_match(engine, settings, req.jd, top_k=req.top_k)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


app = create_app()
