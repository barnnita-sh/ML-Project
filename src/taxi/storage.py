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


def load_tables(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.parquet")) or sorted(folder.glob("*.pkl"))
    read = pd.read_parquet if files and files[0].suffix == ".parquet" else pd.read_pickle
    return pd.concat([read(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
