from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Immutable container for project directory paths."""

    root: Path
    src: Path
    data_root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    notebooks: Path
    models_dir: Path
    reports_dir: Path

    def ensure_directories(self) -> None:
        """Create all data / artifact directories if they do not exist yet."""
        for p in [
            self.data_root,
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.notebooks,
            self.models_dir,
            self.reports_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

DATA_ROOT = ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_INTERIM = DATA_ROOT / "interim"
DATA_PROCESSED = DATA_ROOT / "processed"

NOTEBOOKS = ROOT / "notebooks"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

PATHS = ProjectPaths(
    root=ROOT,
    src=SRC,
    data_root=DATA_ROOT,
    data_raw=DATA_RAW,
    data_interim=DATA_INTERIM,
    data_processed=DATA_PROCESSED,
    notebooks=NOTEBOOKS,
    models_dir=MODELS_DIR,
    reports_dir=REPORTS_DIR,
)

PATHS.ensure_directories()

__all__ = ["ProjectPaths", "PATHS", "ROOT", "SRC"]
