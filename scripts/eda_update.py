"""Recompute the key EDA numbers and charts with the cleaned data (rules v1.0).

  python scripts/eda_update.py

Reads pipeline_output/{empty,trips,snapshots,hexhour}/ and writes
docs/eda_update_v1.md and docs/figures/*.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from taxi import config  # noqa: E402

# reference categorical palette (fixed order), text and grid tokens
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
FIG = config.PROJECT_DIR / "docs" / "figures"


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    return f"figures/{name}"


def read_all(folder, cols, every=1):
    files = sorted(folder.glob("*.parquet")) or sorted(folder.glob("*.pkl"))
    read = pd.read_parquet if files[0].suffix == ".parquet" else pd.read_pickle
    return pd.concat([read(f)[cols].assign(day=f.stem) for f in files[::every]], ignore_index=True)


def km_curve(minutes, event, horizon=180):
    """Kaplan-Meier 'still waiting' curve on a 1-minute grid (censored rows handled)."""
    t = np.minimum(np.ceil(np.asarray(minutes, float)).astype(int), horizon + 1)
    e = np.asarray(event, bool) & (t <= horizon)
    ev = np.bincount(t[e], minlength=horizon + 2)
    out = np.bincount(t, minlength=horizon + 2)
    at_risk = len(t) - np.r_[0, np.cumsum(out)[:-1]]
    s = np.cumprod(1 - np.divide(ev, at_risk, out=np.zeros(len(ev)), where=at_risk > 0))
    return s[: horizon + 1]


def km_median(minutes, event):
    s = km_curve(minutes, event)
    below = np.flatnonzero(s <= 0.5)
    return float(below[0]) if len(below) else float("nan")


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def main():
    out = config.OUT_DIR
    e = read_all(out / "empty", ["t_start", "t_end", "minutes", "pickup_seen", "censor_reason",
                                 "possible_missed_trip", "hex_start"])
    t = read_all(out / "trips", ["t_start", "is_fake"])
    e = e[e["day"] != "20250829"]
    t = t[t["day"] != "20250829"]
    use = e[~e["possible_missed_trip"]].copy()
    use["hour"] = (use["t_start"] // 3600) % 24
    done = use[use["pickup_seen"]]
    lines = ["# EDA update with the cleaned data (rules v1.0)", "",
             "Recomputed after the data audit. Differences from the first EDA come from removing fake "
             "trips, cutting rests out of waits, censoring very long waits and leaving out probable "
             "missed trips. 29 August 2025 (low coverage) is excluded. Waits flagged as possible missed "
             "trips are left out.", ""]

    # 1. distribution of completed waits
    q = done["minutes"].quantile([0.25, 0.5, 0.75, 0.9]).round(1).tolist()
    km_all = km_median(use["minutes"], use["pickup_seen"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(done["minutes"].clip(upper=120), bins=np.arange(0, 122, 2), color=BLUE, edgecolor=SURFACE, lw=0.6)
    ax.axvline(q[1], color=INK2, lw=1, ls="--")
    ax.text(q[1] + 1.5, ax.get_ylim()[1] * 0.9, f"median {q[1]:.0f} min", color=INK2, fontsize=9)
    style(ax, "Completed waits are right-skewed: most under 30 min, a long tail",
          "minutes from drop-off to pickup (120+ shown at 120)", "waits")
    f1 = save(fig, "wait_distribution.png")
    lines += ["## 1. How long waits last", "", f"![Distribution of completed waits]({f1})", "",
              "| Measure | First EDA | Rules v1.0 |", "| --- | --- | --- |",
              f"| Waits analysed | 5,651,362 | {len(use):,} |",
              f"| Censored | 2.0% | {(~use['pickup_seen']).mean():.1%} |",
              f"| Completed wait, 25% / median / 75% / 90% | 10 / 18 / 32 / 54 min | "
              f"{q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} / {q[3]:.0f} min |",
              f"| Median counting censored waits (Kaplan-Meier) | not computed | {km_all:.0f} min |", "",
              "The median of completed waits alone is a little optimistic: censored waits (mostly cut by "
              "a rest) are known to be at least as long as observed, and Kaplan-Meier uses them.", ""]

    # 2. wait by hour of day, and censoring by hour (separate charts, no dual axis)
    byh = done.groupby("hour")["minutes"].median()
    km_h = use.groupby("hour").apply(lambda g: km_median(g["minutes"], g["pickup_seen"]))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(byh.index, byh.values, color=BLUE, lw=2, marker="o", ms=4, label="completed waits only")
    ax.plot(km_h.index, km_h.values, color=ORANGE, lw=2, marker="o", ms=4, label="Kaplan-Meier (with censored)")
    ax.text(23.3, byh.iloc[-1], "completed", color=INK2, fontsize=9, va="center")
    ax.text(23.3, km_h.iloc[-1], "with censored", color=INK2, fontsize=9, va="center")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(-0.5, 26.5)
    style(ax, "Median wait by hour the wait started", "hour of day", "minutes")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    f2 = save(fig, "wait_by_hour.png")

    cen = use.assign(r=use["censor_reason"].where(use["censor_reason"] != "", "pickup"))
    share = pd.crosstab(cen["hour"], cen["r"], normalize="index")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for col, colour, lab in [("off_duty", BLUE, "off duty (rest or fake pickup)"), ("gap", ORANGE, "GPS gap"),
                             ("too_long", AQUA, "over 180 min")]:
        if col in share:
            ax.plot(share.index, share[col] * 100, color=colour, lw=2, label=lab)
    ax.set_xticks(range(0, 24, 3))
    style(ax, "Share of waits censored, by reason and hour the wait started", "hour of day", "% of waits")
    ax.legend(frameon=False, fontsize=9)
    f3 = save(fig, "censoring_by_hour.png")
    lines += ["## 2. Waits and censoring by hour", "", f"![Median wait by hour]({f2})", "",
              f"![Censoring by hour]({f3})", "",
              f"Completed waits are shortest around {int(byh.idxmin()):02d}:00 ({byh.min():.0f} min) and "
              f"longest around {int(byh.idxmax()):02d}:00 ({byh.max():.0f} min). Off-duty censoring peaks at "
              f"{int(share['off_duty'].idxmax()):02d}:00 ({share['off_duty'].max():.0%} of waits starting then), "
              "when many drivers stop for a break or end their shift.", ""]

    # 3. fake trips by hour
    t["hour"] = (t["t_start"] // 3600) % 24
    fk = t.groupby("hour")["is_fake"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(fk.index, fk.values, color=BLUE, width=0.75)
    ax.set_xticks(range(0, 24, 3))
    style(ax, "Share of light-off 'trips' that are really a driver resting", "hour of day", "% of trips")
    f4 = save(fig, "fake_trips_by_hour.png")
    lines += ["## 3. Fake trips", "", f"![Fake trips by hour]({f4})", "",
              f"{t['is_fake'].mean():.1%} of light-off periods are drivers resting with the light switched "
              f"to occupied, from {fk.min():.0f}% at {int(fk.idxmin()):02d}:00 to {fk.max():.0f}% at "
              f"{int(fk.idxmax()):02d}:00. Counting them as pickups would have inflated night-time demand.", ""]

    # 4. busy vs wait per hexagon
    hh = read_all(out / "hexhour", ["hex", "pickups"])
    pk = hh.groupby("hex")["pickups"].sum()
    w = done.groupby("hex_start")["minutes"].agg(["median", "size"])
    w = w[w["size"] >= 100].join(pk, how="inner")
    rho = spearman(w["pickups"], w["median"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(w["pickups"], w["median"], s=9, color=BLUE, alpha=0.35, lw=0)
    ax.set_xscale("log")
    style(ax, f"Busier hexagons are only slightly faster (Spearman {rho:.2f})",
          "real pickups in 2025 (log scale)", "median completed wait, min")
    f5 = save(fig, "busy_vs_wait.png")
    lines += ["## 4. Is the busiest area also the fastest?", "", f"![Busy versus wait]({f5})", "",
              f"Across {len(w):,} hexagons with at least 100 completed waits, the rank correlation between "
              f"yearly real pickups and median wait is {rho:.2f} (first EDA: -0.22 over 2,768 hexagons). "
              "Demand alone still explains little of the wait, which supports predicting waits directly.", ""]

    # 5. remaining wait by minutes already empty (Kaplan-Meier on snapshots)
    s = read_all(out / "snapshots", ["minutes_empty", "minutes_until_end", "pickup_seen"], every=7)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    rows = []
    for m, colour in [(0, BLUE), (15, ORANGE), (30, AQUA), (60, YELLOW)]:
        g = s[s["minutes_empty"] == m]
        curve = km_curve(g["minutes_until_end"], g["pickup_seen"], horizon=120)
        ax.plot(np.arange(121), curve * 100, color=colour, lw=2, label=f"already empty {m} min")
    for m in [0, 5, 15, 30, 45, 60, 90]:
        g = s[s["minutes_empty"] == m]
        c = km_curve(g["minutes_until_end"], g["pickup_seen"], horizon=120)
        rows.append((m, len(g), km_median(g["minutes_until_end"], g["pickup_seen"]), 100 - c[15] * 100))
    style(ax, "Chance of still waiting, by how long the taxi has already been empty",
          "minutes from now", "% still waiting")
    ax.legend(frameon=False, fontsize=9)
    f6 = save(fig, "survival_by_minutes_empty.png")
    lines += ["## 5. Survival curves: the wait ahead depends on the wait so far", "",
              f"![Survival curves]({f6})", "",
              "Kaplan-Meier on Model B snapshots (every 7th day), censored snapshots included:", "",
              "| Already empty | Snapshots | Median further wait | Chance of a pickup within 15 min |",
              "| --- | --- | --- | --- |"]
    for m, n, med, p15 in rows:
        lines.append(f"| {m} min | {n:,} | {med:.0f} min | {p15:.0f}% |")
    lines += ["", "This is the historical survival baseline in its simplest form (one curve per "
              "minutes-empty value, no area or hour). Model B must beat curves like these.", ""]

    # 6. daily real pickups
    daily = t[~t["is_fake"]].groupby("day").size()
    daily.index = pd.to_datetime(daily.index)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(daily.index, daily.values, color=BLUE, lw=1.6)
    style(ax, "Real pickups per day, 2025", "", "pickups")
    f7 = save(fig, "daily_pickups.png")
    lines += ["## 6. Across the year", "", f"![Daily pickups]({f7})", "",
              f"Median {int(daily.median()):,} real pickups a day (range {int(daily.min()):,} to "
              f"{int(daily.max()):,}).", ""]

    dst = config.PROJECT_DIR / "docs" / "eda_update_v1.md"
    dst.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
