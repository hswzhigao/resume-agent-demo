"""配置：环境变量 / .env；密钥只从环境读取，不写入代码或日志。"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5433/resumes"

    # Embedding：OpenAI 兼容接口；不配置则用本地 hash 向量（256 维）
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    embedding_dim: int = 1536
    local_embedding_dim: int = 256

    # Chat：可选，用于查询改写 / 评分 / 解释
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # Rerank：可选，Jina/Cohere 风格
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = ""
    rerank_path: str = "/rerank"

    mask_pii: bool = True

    @property
    def has_embedding(self) -> bool:
        return bool(self.embedding_base_url and self.embedding_api_key and self.embedding_model)

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key and self.llm_model)

    @property
    def has_rerank(self) -> bool:
        return bool(self.rerank_base_url and self.rerank_api_key and self.rerank_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()
