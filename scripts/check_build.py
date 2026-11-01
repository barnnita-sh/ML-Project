"""Check that a build is complete and consistent before using it.

  python scripts/check_build.py              # checks pipeline_output/
  python scripts/check_build.py --features   # also checks hexhour/ and features/

Passes only if every expected day exists in every table, all manifests have the
current rules version and real H3 hexagons, and each folder holds one file format.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from taxi import config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="20250101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--features", action="store_true")
    a = p.parse_args()
    days = {d.strftime("%Y%m%d") for d in pd.date_range(a.start, a.end)}
    days &= {f.stem for f in config.TAXI_DAYS_DIR.glob("*.pkl")} or days
    out, problems = config.OUT_DIR, []

    folders = ["trips", "empty", "snapshots", "manifest"]
    if a.features:
        folders += ["hexhour", "features/snapshots", "features/modelA"]
    for f in folders:
        files = [x for x in (out / f).glob("*") if x.is_file() and not x.name.endswith(".tmp")]
        kinds = {x.suffix for x in files}
        got = {x.stem for x in files}
        if len(kinds) > 1:
            problems.append(f"{f}: mixed file formats {sorted(kinds)}; move the old ones out")
        missing = sorted(days - got)
        if missing:
            problems.append(f"{f}: {len(missing)} days missing (first: {missing[:3]})")
        print(f"{f:20s} {len(got & days):4d}/{len(days)} days, format {sorted(kinds)}")

    man = [json.loads(x.read_text()) for x in sorted((out / "manifest").glob("*.json"))]
    rules = pd.Series([m.get("rules_version") for m in man]).value_counts().to_dict()
    grids = pd.Series([m.get("grid") for m in man]).value_counts().to_dict()
    print("rules versions:", rules)
    print("grids:", grids)
    if set(rules) != {config.RULES_VERSION}:
        problems.append(f"manifests use rules {rules}, expected only {config.RULES_VERSION}")
    if set(grids) != {"h3"}:
        problems.append(f"manifests use grids {grids}; rebuild with h3 installed (--force)")

    if problems:
        print("\nPROBLEMS:")
        for x in problems:
            print(" -", x)
        sys.exit(1)
    print("\nOK: build is complete and consistent")


if __name__ == "__main__":
    main()
