from dataclasses import dataclass
import os

from dotenv import load_dotenv

from .paths import ROOT

load_dotenv(ROOT / ".env", override=False)


@dataclass
class Settings:
    environment: str = os.getenv("ENVIRONMENT", "dev")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))

    date_column: str = os.getenv("DATE_COLUMN", "date")
    target_column: str = os.getenv("TARGET_COLUMN", "log_price")
    cocoa_freq: str = os.getenv("COCOA_FREQ", "B")

    data_dir: str = os.getenv("DATA_DIR", "data")
    raw_subdir: str = os.getenv("RAW_SUBDIR", "raw")
    interim_subdir: str = os.getenv("INTERIM_SUBDIR", "interim")
    processed_subdir: str = os.getenv("PROCESSED_SUBDIR", "processed")

    notebooks_subdir: str = os.getenv("NOTEBOOKS_SUBDIR", "notebooks")
    models_subdir: str = os.getenv("MODELS_SUBDIR", "models")
    reports_subdir: str = os.getenv("REPORTS_SUBDIR", "reports")


settings = Settings()

__all__ = ["Settings", "settings"]
