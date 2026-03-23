#!/usr/bin/env python3
"""Terminal entry: Adaptive RFS family tracker."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking_core.tracker_cli import main_for

if __name__ == "__main__":
    raise SystemExit(main_for("Adaptive RFS family"))
