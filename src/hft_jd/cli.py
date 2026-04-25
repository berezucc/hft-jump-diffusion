"""CLI entry points."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_dashboard() -> None:
    """Launch the Streamlit dashboard."""
    repo_root = Path(__file__).resolve().parents[2]
    app = repo_root / "dashboard" / "app.py"
    if not app.exists():
        sys.stderr.write(f"Dashboard not found at {app}\n")
        sys.exit(1)
    os.execvp("streamlit", ["streamlit", "run", str(app)])


if __name__ == "__main__":
    run_dashboard()
