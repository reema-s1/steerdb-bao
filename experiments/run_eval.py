"""Regenerate every table and figure in docs/results.md from the experience store.

    python experiments/run_eval.py                # full evaluation + ablations
    python experiments/run_eval.py --overhead     # also measure planning/inference overhead (needs DB)

Figures are written to docs/figures/ so the results page is self-contained in the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steerdb import config  # noqa: E402
from steerdb.cli import main  # noqa: E402

if __name__ == "__main__":
    defaults = ["bench", "--out", str(config.REPO_ROOT / "docs")]
    main(defaults + sys.argv[1:])
