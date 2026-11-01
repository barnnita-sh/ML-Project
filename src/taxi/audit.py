"""Pick empty periods for the manual audit and draw one picture per period.

Both labellers look at the SAME traces, label them separately, then compare.
The point is to check that our rules (light = occupancy, gaps, breaks,
off-duty) match what the GPS trace actually shows, before freezing them.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

# (stratum name, rule, how many)
STRATA = [
    ("night_0_to_5",     lambda e: e["hour"].between(0, 4), 8),
    ("very_short_lt3",   lambda e: e["minutes"] < 3, 7),
    ("long_60_plus",     lambda e: e["minutes"] >= 60, 8),
    ("rest_or_too_long", lambda e: e["censor_reason"].isin(["off_duty", "too_long"]), 7),
    ("gap_censored",     lambda e: e["censor_reason"].eq("gap"), 6),
    ("mostly_engine_off", lambda e: e["engine_off_share"] > 0.5, 7),
    ("random",           lambda e: pd.Series(True, index=e.index), 7),
]

LABELS = ["searching", "waiting_at_spot", "off_duty", "light_error_or_missed_trip", "gps_problem", "unsure"]


def sample_for_audit(empty: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    e = empty.copy()
    e["hour"] = (e["t_start"] // 3600) % 24
    picked, used = [], set()
    rng = np.random.default_rng(seed)
    for name, rule, n in STRATA:
        pool = e[rule(e) & ~e["episode_id"].isin(used)]
        take = pool.iloc[rng.permutation(len(pool))[:n]].assign(stratum=name)
        used.update(take["episode_id"])
        picked.append(take)
    return pd.concat(picked, ignore_index=True)


def plot_episode(d: pd.DataFrame, ep, path):
    """Map of the trace (left) and speed / engine / light over time (right)."""
    pad = config.AUDIT_CONTEXT_MIN * 60
    g = d[(d["vid"] == ep.vid) & d["t"].between(ep.t_start - pad, ep.t_end + pad)].copy()
    inside = g["t"].between(ep.t_start, ep.t_end).to_numpy()
    mins = ((g["t"] - ep.t_start) / 60).to_numpy()
    end_min = (ep.t_end - ep.t_start) / 60

    # positions in km from the drop-off point (east, north), so distances are readable
    km_lon = 111.320 * np.cos(np.radians(ep.lat_start))
    x = ((g["lon"] - ep.lon_start) * km_lon).to_numpy()
    y = ((g["lat"] - ep.lat_start) * 110.574).to_numpy()
    xe, ye = (ep.lon_end - ep.lon_start) * km_lon, (ep.lat_end - ep.lat_start) * 110.574

    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.5], height_ratios=[3, 0.7, 0.7], hspace=0.15, wspace=0.18)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_spd = fig.add_subplot(gs[0, 1])
    ax_eng = fig.add_subplot(gs[1, 1], sharex=ax_spd)
    ax_lgt = fig.add_subplot(gs[2, 1], sharex=ax_spd)

    # ---------------- left: map
    ax_map.plot(x, y, color="0.8", lw=1, zorder=1)
    ax_map.scatter(x[~inside], y[~inside], s=10, c="0.65", label="10 min before / after")
    ax_map.scatter(x[inside], y[inside], s=14, zorder=3,
                   c=np.where(g["occ"].to_numpy()[inside] == 0, "tab:green", "tab:red"),
                   label="during the period (green = empty)")
    ax_map.scatter([0], [0], marker="^", s=110, c="k", zorder=4, label="drop-off (start)")
    ax_map.scatter([xe], [ye], marker="s", s=80, c="tab:blue", zorder=4, label="end (pickup or censored)")
    half = max(1.0, np.nanmax(np.abs(np.r_[x, y])) * 1.1)            # show at least 2 km x 2 km
    ax_map.set_xlim(-half, half)
    ax_map.set_ylim(-half, half)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("km east of drop-off (negative = west)", fontsize=8)
    ax_map.set_ylabel("km north of drop-off (negative = south)", fontsize=8)
    ax_map.grid(alpha=0.3)
    ax_map.tick_params(labelsize=7)
    ax_map.legend(fontsize=7, loc="upper left")
    xi, yi = x[inside], y[inside]
    km_driven = float(np.sum(np.hypot(np.diff(xi), np.diff(yi)))) if len(xi) > 1 else 0.0
    ax_map.text(0.02, 0.02,
                f"during the period:\n{inside.sum()} GPS pings (pings at one spot overlap)\n"
                f"{km_driven:.1f} km driven\n"
                f"furthest {np.max(np.hypot(xi, yi)) if len(xi) else 0:.1f} km from drop-off\n"
                f"longest stop with engine off: {ep.longest_still_min:.0f} min",
                transform=ax_map.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    # ---------------- right: three timelines sharing the time axis
    for ax in (ax_spd, ax_eng, ax_lgt):
        ax.axvspan(0, end_min, color="khaki", alpha=0.25, lw=0)       # the period itself
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.axvline(end_min, color="tab:blue", ls="--", lw=0.8)
        ax.tick_params(labelsize=7)
    ax_spd.plot(mins, g["speed"], ".-", lw=1, ms=3, color="tab:blue")
    ax_spd.set_ylabel("speed (km/h)", fontsize=8)
    ax_spd.set_ylim(bottom=0)
    ax_spd.text(0, 1.02, "drop-off", transform=ax_spd.get_xaxis_transform(), fontsize=7, ha="center")
    ax_spd.text(end_min, 1.02, "end", transform=ax_spd.get_xaxis_transform(), fontsize=7, ha="center",
                color="tab:blue")
    plt.setp(ax_spd.get_xticklabels(), visible=False)

    def strip(ax, state, colours, labels):
        """Each ping's state is drawn from that ping until the next one."""
        widths = np.diff(mins)
        for value, colour, label in zip((1, 0), colours, labels):
            segs = [(m, w) for m, w, s in zip(mins[:-1], widths, state[:-1]) if s == value]
            ax.broken_barh(segs, (0, 1), color=colour, alpha=0.7, label=label)
        ax.set_ylim(0, 1)

    acc = g["acc"].to_numpy()
    strip(ax_eng, acc, ("tab:olive", "0.55"), ("on", "off"))
    ax_eng.set_yticks([])
    ax_eng.set_ylabel("engine", fontsize=8, rotation=0, ha="right", va="center")
    ax_eng.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.setp(ax_eng.get_xticklabels(), visible=False)

    occ = g["occ"].to_numpy()
    strip(ax_lgt, occ, ("tab:red", "tab:green"), ("occupied", "empty"))
    if "occ_raw" in g:
        fixed = (g["occ_raw"] != g["occ"]).to_numpy()
        ax_lgt.scatter(mins[fixed], np.full(fixed.sum(), 0.5), marker="x", c="k", s=25, zorder=3,
                       label="flicker fixed")
    ax_lgt.set_yticks([])
    ax_lgt.set_ylabel("light", fontsize=8, rotation=0, ha="right", va="center")
    ax_lgt.set_xlabel("minutes from drop-off (negative = before the drop-off)", fontsize=8)
    ax_lgt.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))

    start = pd.to_datetime(ep.t_start, unit="s").strftime("%a %d %b %Y %H:%M")
    ended = "pickup" if ep.pickup_seen else f"censored ({ep.censor_reason})"
    fig.suptitle(f"{ep.stratum}  |  {start}  |  empty for {ep.minutes:.1f} min  |  ended by: {ended}\n"
                 f"{ep.episode_id}", fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.92, top=0.88, bottom=0.1)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def labelling_sheet(sample: pd.DataFrame, plot_dir: str = "plots") -> pd.DataFrame:
    s = sample.copy()
    s["day"] = pd.to_datetime(s["t_start"], unit="s").dt.strftime("%Y%m%d")
    s["start_local"] = pd.to_datetime(s["t_start"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
    s["plot_file"] = plot_dir + "/" + s["episode_id"].str.replace("[^A-Za-z0-9_-]", "_", regex=True) + ".png"
    cols = ["episode_id", "stratum", "day", "start_local", "minutes", "pickup_seen", "censor_reason",
            "engine_off_share", "longest_still_min", "start_type", "possible_missed_trip", "plot_file"]
    s = s[cols].round(2)
    for c in ["label_A", "label_B", "agreed_label", "notes"]:
        s[c] = ""
    return s
