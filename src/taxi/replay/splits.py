"""Replay day selection and snapshot decision points.

Reuses the project's authoritative date definitions.  No random splitting.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from taxi import config
from taxi.models.data import EXCLUDED_DAYS, SPLITS, split_of
from taxi.storage import load_day_table

DECISION_MINUTES = (0, 5, 15, 30, 45)


def replay_days(split: str, folder=None) -> list[str]:
    """Return available YYYYMMDD snapshot days in the chosen validation/test split."""
    if split not in ("val", "test"):
        raise ValueError("Replay is only allowed on 'val' or 'test' (not train).")
    folder = Path(folder) if folder is not None else config.OUT_DIR / "features" / "snapshots"
    found = {p.stem for p in folder.iterdir() if p.suffix in (".pkl", ".parquet")} if folder.exists() else set()
    return sorted(day for day in found if day not in EXCLUDED_DAYS and split_of(day) == split)


def load_replay_day(day: str, split: str, folder=None) -> pd.DataFrame:
    """Load a daily snapshot table only if it belongs to the requested split."""
    if split not in ("val", "test") or split_of(day) != split:
        raise ValueError(f"Day {day} is not permitted in replay split {split!r}.")
    folder = Path(folder) if folder is not None else config.OUT_DIR / "features" / "snapshots"
    df = load_day_table(folder, day)
    if df is None:
        raise FileNotFoundError(f"No snapshot table for {day} in {folder}")
    return df


def decision_snapshots(df: pd.DataFrame, moments=DECISION_MINUTES) -> pd.DataFrame:
    """Select one decision per episode at each desired minutes-empty checkpoint.

    The pipeline samples every five minutes.  A checkpoint must match its
    observed snapshot to within 2.5 minutes, and one episode/checkpoint can
    contribute at most one row.  No interpolation or future row is created.
    """
    required = {"episode_id", "snapshot_t", "minutes_empty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Snapshot table lacks columns: {sorted(missing)}")
    if df.empty:
        return df.copy().assign(decision_minute=pd.Series(dtype="int64"))
    parts = []
    minutes = pd.to_numeric(df["minutes_empty"], errors="coerce")
    for m in moments:
        sub = df.loc[(minutes - m).abs() <= 2.5].copy()
        if sub.empty:
            continue
        sub["_error"] = (pd.to_numeric(sub["minutes_empty"]) - m).abs()
        sub = sub.sort_values(["_error", "snapshot_t"]).drop_duplicates("episode_id")
        sub["decision_minute"] = m
        parts.append(sub.drop(columns="_error"))
    if not parts:
        return df.iloc[:0].copy().assign(decision_minute=pd.Series(dtype="int64"))
    return pd.concat(parts, ignore_index=True).sort_values(["snapshot_t", "episode_id", "decision_minute"]).reset_index(drop=True)
