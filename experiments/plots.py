"""Redraw figures from an existing results.json without rerunning the evaluation.

python experiments/plots.py [docs/results.json] [docs/figures]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steerdb.plots import make_all  # noqa: E402

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "docs" / "results.json"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else results.parent / "figures"
    for p in make_all(json.loads(results.read_text()), out):
        print(p)
