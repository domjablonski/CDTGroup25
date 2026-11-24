"""Read the shared Food.com inputs and stage generated results before publishing."""

import ast
from collections.abc import Iterator
from contextlib import contextmanager
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

DEFAULT_SEED = 42
RECIPE_ID_COLUMNS = ("recipe_id", "id")
CSV_ID_COLUMNS = (*RECIPE_ID_COLUMNS, "canonical_id")
RecipeId = int | str


def read_table(path: Path, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> pd.DataFrame:
    """Load only the requested CSV or Parquet columns."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        available = pd.read_csv(path, nrows=0).columns
    elif suffix == ".parquet":
        from pyarrow.parquet import read_schema

        available = read_schema(path).names
    else:
        raise ValueError(f"Expected a .csv or .parquet file, got {path}")

    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    columns = list(dict.fromkeys((*required, *(col for col in optional if col in available))))

    if suffix == ".csv":
        id_dtypes = {column: "string" for column in columns if column in CSV_ID_COLUMNS}
        return pd.read_csv(path, usecols=columns, dtype=id_dtypes)
    return pd.read_parquet(path, columns=columns)


def parse_ingredients(value: object) -> tuple[str, ...]:
    """Parse CSV list literals and Parquet arrays without changing ingredient names."""
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return ()
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Expected an ingredient list, got {value!r}") from exc
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Expected a list of ingredient strings, got {value!r}")
    return tuple(value)


def read_recipes(
    path: Path,
    ingredient_col: str,
    sample_size: int | None = None,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Read an optional seeded subset, preserving the source order in that subset."""
    if sample_size is not None and sample_size < 1:
        raise ValueError("Sample size must be positive.")
    recipes = read_table(path, (ingredient_col,), (*RECIPE_ID_COLUMNS, "name"))
    if recipes.empty:
        raise ValueError(f"{path} contains no recipes.")
    if sample_size is not None and sample_size < len(recipes):
        recipes = recipes.sample(n=sample_size, random_state=seed).sort_index()
    recipes[ingredient_col] = recipes[ingredient_col].map(parse_ingredients)
    return recipes


def parse_recipe_id(value: object) -> RecipeId:
    """Use the same ID for a CSV integer string and its Parquet integer."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and value.strip():
        value = value.strip()
        try:
            number = int(value)
        except ValueError:
            return value
        return number if str(number) == value else value
    raise ValueError(f"Expected a nonempty recipe ID, got {value!r}")


def recipe_ids(recipes: pd.DataFrame) -> list[RecipeId]:
    column = next((name for name in RECIPE_ID_COLUMNS if name in recipes), None)
    if column is None:
        raise ValueError("Recipes must have a recipe_id or id column.")
    ids = [parse_recipe_id(value) for value in recipes[column]]
    if len(set(ids)) != len(ids):
        raise ValueError("Recipe IDs must be unique.")
    return ids


@contextmanager
def staged_outputs(out_dir: Path) -> Iterator[Path]:
    """Publish complete files; publish a JSON handoff manifest after its Parquets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".foodcom-", dir=out_dir) as directory:
        stage = Path(directory)
        yield stage
        for file in sorted(stage.iterdir(), key=lambda path: (path.suffix == ".json", path.name)):
            file.replace(out_dir / file.name)
