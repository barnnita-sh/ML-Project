"""Save and load tables. Parquet when pyarrow is installed, pickle otherwise."""
import json
from pathlib import Path

import pandas as pd

try:
    import pyarrow  # noqa: F401
    FORMAT = "parquet"
except ImportError:  # pragma: no cover
    FORMAT = "pkl"


def save_table(df: pd.DataFrame, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    for c in df.columns:                      # h3 IDs are objects; parquet wants plain strings
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    path = folder / f"{name}.{FORMAT}"
    tmp = path.with_suffix(path.suffix + ".tmp")
    if FORMAT == "parquet":
        df.to_parquet(tmp, index=False)
    else:
        df.to_pickle(tmp)
    tmp.replace(path)
    return path


def _plain_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet (with pandas 3) reads text as the 'string' dtype while pickles and freshly built
    tables use plain Python objects. Mixing them breaks merges, so always use objects."""
    for c in df.columns:
        if isinstance(df[c].dtype, pd.StringDtype):
            df[c] = df[c].astype(object)
    return df


def load_tables(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.parquet")) or sorted(folder.glob("*.pkl"))
    read = pd.read_parquet if files and files[0].suffix == ".parquet" else pd.read_pickle
    return _plain_strings(pd.concat([read(f) for f in files], ignore_index=True)) if files else pd.DataFrame()


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_day_table(folder: Path, day: str):
    """One day's table from a pipeline folder, or None if it does not exist."""
    for suffix, read in ((".parquet", pd.read_parquet), (".pkl", pd.read_pickle)):
        f = folder / f"{day}{suffix}"
        if f.exists():
            return _plain_strings(read(f))
    return None
