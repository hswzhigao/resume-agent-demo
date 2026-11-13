"""共享 fixtures：真实 pgvector（Testcontainers 或 CI 注入 DATABASE_URL）。"""

import os

import pytest

try:
    import testcontainers.postgres  # noqa: F401

    HAS_TESTCONTAINERS = True
except ImportError:
    HAS_TESTCONTAINERS = False


@pytest.fixture(scope="session")
def pg_url():
    """优先级：CI 注入 DATABASE_URL → pgserver 嵌入式 → Testcontainers（显式 opt-in）。

    pgserver 是自带 pgvector 的嵌入式 PostgreSQL（pip 预编译），本地零依赖。
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        yield url
        return
    try:
        import pgserver

        default_dir = os.path.expanduser("~/.cache/resume-agent-demo/pgdata")
        data_dir = os.environ.get("PGSERVER_DATA_DIR", default_dir)
        server = pgserver.get_server(data_dir)
        uri = server.get_uri()
        yield uri.replace("postgresql://", "postgresql+psycopg://", 1)
        return
    except Exception as e:  # noqa: BLE001
        print(f"\n[pgserver 不可用: {e}]")

    if os.environ.get("PGVECTOR_IMAGE"):
        from testcontainers.postgres import PostgresContainer

        with PostgresContainer(os.environ["PGVECTOR_IMAGE"], port=5432) as pg:
            yield pg.get_connection_url(driver="psycopg")
        return
    pytest.skip(
        "no pg backend available: set DATABASE_URL, install pgserver, or set PGVECTOR_IMAGE"
    )


@pytest.fixture(scope="session")
def engine(pg_url):
    from sqlalchemy import create_engine

    return create_engine(pg_url)


@pytest.fixture(scope="session")
def settings():
    from resume_agent_demo.config import Settings

    return Settings(mask_pii=False)


@pytest.fixture(scope="session")
def mask_settings():
    from resume_agent_demo.config import Settings

    return Settings(mask_pii=True)
