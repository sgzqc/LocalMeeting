from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models" / "sherpa-onnx-zipformer"
UPLOAD_DIR = ROOT_DIR / "uploads"


@dataclass(frozen=True)
class Settings:
    openai_base_url: str
    openai_api_key: str
    openai_model_name: str
    summary_interval_seconds: int = 10
    sample_rate: int = 16000

    @property
    def llm_ready(self) -> bool:
        return bool(self.openai_api_key and self.openai_model_name)


def get_settings() -> Settings:
    load_dotenv(ROOT_DIR / ".env", override=False)
    return Settings(
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model_name=os.getenv("OPENAI_MODEL_NAME", ""),
    )
