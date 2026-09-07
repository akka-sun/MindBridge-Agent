from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class RagPipelineSettings(BaseSettings):
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_embedding_model: str = "BAAI/bge-m3"
    siliconflow_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rag_experiment_dir: str = "data/rag-experiments"
    rag_embedding_batch_size: int = 32
    rag_request_timeout_seconds: float = 120.0
    rag_request_max_retries: int = 5
    rag_rrf_k: int = 60
    rag_dense_candidates: int = 30
    rag_bm25_candidates: int = 30
    rag_rerank_candidates: int = 30
    rag_final_top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def experiment_root(self) -> Path:
        value = Path(self.rag_experiment_dir)
        return value if value.is_absolute() else self.project_root / value


@lru_cache
def get_rag_settings() -> RagPipelineSettings:
    return RagPipelineSettings()

