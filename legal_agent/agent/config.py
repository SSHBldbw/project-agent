import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    cwd = Path.cwd()
    candidates = [
        Path(r"d:\AI_\.env"),
        Path(r"d:\AI_\test\legal_agent\.env"),
        cwd / ".env",
        cwd.parent / ".env",
        cwd.parent.parent / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            break


load_env()

_DASHSCOPE_KEY = os.getenv("DASHSCOPE_API_KEY", "")
_ALI_KEY = os.getenv("ALI_API_KEY", "")
if not _DASHSCOPE_KEY and _ALI_KEY:
    os.environ["DASHSCOPE_API_KEY"] = _ALI_KEY


@dataclass
class AgentConfig:
    llm_model: str = "qwen3-max"
    llm_api_key: str = os.getenv("DASHSCOPE_API_KEY", "") or os.getenv("ALI_API_KEY", "")

    baidu_api_key: str = os.getenv("BAIDU_API_KEY", "")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")

    persist_directory: str = os.getenv("LAW_CHROMA_PERSIST_DIR", ".chroma_legal")
    top_k: int = 12

    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay: float = float(os.getenv("RETRY_DELAY", "1.0"))
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "3"))
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "1048576"))


def validate_key(cfg: AgentConfig) -> None:
    if not cfg.llm_api_key:
        raise ValueError("缺少 DASHSCOPE_API_KEY，请先配置环境变量。")


def get_config() -> AgentConfig:
    return AgentConfig()
