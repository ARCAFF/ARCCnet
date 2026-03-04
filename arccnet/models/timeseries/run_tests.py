#!/usr/bin/env python3
"""Run timeseries tests via pytest."""

import sys
import subprocess
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    cmd = [sys.executable, "-m", "pytest", "arccnet/models/timeseries"]
    return subprocess.call(cmd, cwd=project_root)


if __name__ == "__main__":
    raise SystemExit(main())
