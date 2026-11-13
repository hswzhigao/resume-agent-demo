"""FastAPI 集成测试（真实 pgvector）。"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from resume_agent_demo.api import create_app

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture(scope="module")
def client(engine, settings):
    return TestClient(create_app(settings=settings, engine=engine))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["embedding"] == "local-hash"


def test_ingest_search_match_full_flow(client):
    r = client.post("/ingest", json={"folder": str(SAMPLE_DIR), "reset": True})
    assert r.status_code == 200
    body = r.json()
    assert body["docs"] == 3  # 3 份合成简历（JD 单独放 data/jd，不进库）
    assert body["chunks"] >= 6

    r = client.post("/search", json={"query": "Java 高并发", "top_k": 3})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results
    assert "Java" in results[0]["content"]

    jd_dir = Path(__file__).resolve().parents[1] / "data" / "jd"
    jd = (jd_dir / "sample_jd.md").read_text(encoding="utf-8")
    r = client.post("/match", json={"jd": jd, "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] in {"llm", "heuristic"}
    assert "explanation" in body


def test_search_validation_error(client):
    r = client.post("/search", json={"query": "", "top_k": 1})
    assert r.status_code == 422


def test_ingest_missing_folder(client):
    r = client.post("/ingest", json={"folder": "/nonexistent/dir"})
    assert r.status_code == 400
