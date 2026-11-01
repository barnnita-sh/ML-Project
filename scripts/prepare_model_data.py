"""Build the model training, validation and test tables from the feature tables.

  python scripts/prepare_model_data.py                       # default sample sizes
  python scripts/prepare_model_data.py --frac-b-train 0.05   # smaller, for a quick run

Splits by date: warm-up 1-28 Jan and 29 Aug left out; train 29 Jan - 15 Sep;
validation 16 Sep - 9 Nov; test 10 Nov - 31 Dec. Each wait stays in one split (all its
snapshots are in the file of the day it started). Model B rows whose outcome window crosses
into the next split are dropped from train and validation (purging).
Sampling is deterministic: Model B samples whole waits, Model A samples hexagon-hours.

Writes pipeline_output/model_input/{modelA,modelB}_{train,val,test}.* and model_input/info.json
The test tables are written now but must not be looked at until the final test run.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taxi import config  # noqa: E402
from taxi.models.data import MODEL_INPUT, prepare  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frac-a-train", type=float, default=0.25)
    p.add_argument("--frac-a-val", type=float, default=0.5)
    p.add_argument("--frac-a-test", type=float, default=0.5)
    p.add_argument("--frac-b-train", type=float, default=0.12)
    p.add_argument("--frac-b-val", type=float, default=0.2)
    p.add_argument("--frac-b-test", type=float, default=0.2)
    a = p.parse_args()
    fa = {"train": a.frac_a_train, "val": a.frac_a_val, "test": a.frac_a_test}
    fb = {"train": a.frac_b_train, "val": a.frac_b_val, "test": a.frac_b_test}
    info = prepare(fa, fb, verbose=False)
    info["fractions"] = {"modelA": fa, "modelB": fb}
    dst = config.OUT_DIR / MODEL_INPUT / "info.json"
    dst.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
