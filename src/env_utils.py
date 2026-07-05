import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_environment() -> Path:
    """Load environment variables from a local .env file if present."""
    project_root = Path(__file__).resolve().parent.parent
    dotenv_candidates = [project_root / ".env", Path.cwd() / ".env"]

    for dotenv_path in dotenv_candidates:
        if dotenv_path.exists():
            load_dotenv(dotenv_path, override=False)
            break
    else:
        load_dotenv(override=False)

    return project_root


def get_required_env(name: str, default: Optional[str] = None) -> str:
    """Return an environment variable or raise a clear deployment error."""
    load_environment()
    value = os.getenv(name, default)
    if value is None or str(value).strip() == "":
        project_root = Path(__file__).resolve().parent.parent
        raise ValueError(
            f"{name} is missing. Set it in your shell, Railway Variables, or in {project_root / '.env'}."
        )
    return str(value).strip()
