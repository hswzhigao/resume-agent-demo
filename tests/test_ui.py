"""测试：GET / 返回操作页面 HTML。"""

import pytest
from fastapi.testclient import TestClient

from resume_agent_demo.api import create_app
from resume_agent_demo.config import Settings


@pytest.fixture
def client():
    return TestClient(create_app(settings=Settings(mask_pii=True), engine=object()))


def test_root_serves_html_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "简历" in body
    assert "match" in body and "search" in body


def test_root_page_mentions_synthetic_data_notice(client):
    r = client.get("/")
    assert "合成" in r.text
