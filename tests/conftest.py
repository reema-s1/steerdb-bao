import copy
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def plan_3way() -> dict:
    return json.loads((FIXTURES / "plan_3way.json").read_text())


def make_plan(join: str = "Hash Join", rows: float = 1000, cost: float = 5000) -> dict:
    """A small synthetic plan whose join operator/row estimate we control."""
    return {
        "Node Type": "Aggregate",
        "Total Cost": cost + 10,
        "Plan Rows": 1,
        "Plan Width": 8,
        "Plans": [
            {
                "Node Type": join,
                "Total Cost": cost,
                "Plan Rows": rows,
                "Plan Width": 16,
                "Plans": [
                    {
                        "Node Type": "Seq Scan",
                        "Relation Name": "title",
                        "Total Cost": cost / 2,
                        "Plan Rows": rows * 2,
                        "Plan Width": 8,
                    },
                    {
                        "Node Type": "Index Scan",
                        "Relation Name": "cast_info",
                        "Total Cost": cost / 4,
                        "Plan Rows": rows,
                        "Plan Width": 8,
                        "Index Cond": "(x = y)",
                    },
                ],
            }
        ],
    }


@pytest.fixture
def plan_factory():
    return lambda **kw: copy.deepcopy(make_plan(**kw))
