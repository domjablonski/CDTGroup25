"""Shared test fixtures (FROZEN: docs/CONTRACT.md 3.5).

mini_recipes / mini_interactions: 8 recipes and 12 reviews in the raw Kaggle format
(tests/fixtures/), cleaned with foodcom.cleaning. They contain on purpose: a copy (102/103),
the overflow duration, 0 minutes, > 5,000 kcal, a rating 0, a duplicate review, an unknown
recipe and a self-review. CI only has these files: tests that need the real data are marked
@pytest.mark.data and are skipped when data/processed/recipes.parquet is missing.
"""
from pathlib import Path

import pandas as pd
import pytest

from foodcom import cleaning, config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def mini_recipes() -> pd.DataFrame:
    return cleaning.clean_recipes(pd.read_csv(FIXTURES / "RAW_recipes_mini.csv"))


@pytest.fixture(scope="session")
def mini_interactions(mini_recipes) -> pd.DataFrame:
    raw = pd.read_csv(FIXTURES / "RAW_interactions_mini.csv")
    return cleaning.clean_interactions(raw, mini_recipes["recipe_id"])


def pytest_runtest_setup(item):
    if item.get_closest_marker("data") and not config.RECIPES.exists():
        pytest.skip("needs the real data (python -m foodcom.cleaning)")
