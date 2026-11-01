"""Hexagons and distances.

Uses the real `h3` package (resolution 8) when installed. If it is not, falls
back to a home-made hexagon grid of the same cell size so the code still runs,
but the IDs are NOT H3 IDs. Every output records which grid was used in GRID.
Install h3 on your machine:  pip install "h3>=4"
"""
import warnings

import numpy as np

from . import config

try:
    import h3

    GRID = "h3"
except ImportError:  # pragma: no cover
    h3 = None
    GRID = "fallback-hexgrid"
    warnings.warn("h3 is not installed: using the stand-in hexagon grid. "
                  "Run `pip install h3` before building real outputs.")

# stand-in grid constants (same as the EDA notebook)
_LAT0, _LON0 = 13.75, 100.50
_KM_LAT, _KM_LON = 110.574, 111.320 * np.cos(np.radians(_LAT0))
_SIDE = np.sqrt(2 * 0.737 / (3 * np.sqrt(3)))


def to_hex(lat, lon):
    """Hexagon ID for each point. Strings for h3, integers for the fallback grid."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if h3 is not None:
        return np.array([h3.latlng_to_cell(a, b, config.H3_RES) for a, b in zip(lat, lon)], dtype=object)
    x, y = (lon - _LON0) * _KM_LON, (lat - _LAT0) * _KM_LAT
    q, r = (np.sqrt(3) / 3 * x - y / 3) / _SIDE, (2 / 3 * y) / _SIDE
    s = -q - r
    rq, rr, rs = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    rq = np.where(fix_q, -rr - rs, rq)
    rr = np.where(fix_r, -rq - rs, rr)
    return (rq.astype(np.int64) + 5000) * 10000 + (rr.astype(np.int64) + 5000)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# axial-coordinate offsets of the 6 neighbours in the stand-in grid's integer IDs
_FALLBACK_NBR_OFFSETS = np.array([0, 10000, -10000, 1, -1, 9999, -9999], dtype=np.int64)


def neighbours(hexes) -> "pd.DataFrame":
    """Pairs (hex, nbr) for each hexagon and itself plus its 6 neighbours (7 rows per hexagon)."""
    import pandas as pd
    hexes = pd.unique(np.asarray(hexes))
    if len(hexes) and isinstance(hexes[0], str):
        if h3 is None:
            raise RuntimeError("H3 IDs found but h3 is not installed")
        rows = [(h, n) for h in hexes for n in h3.grid_disk(h, 1)]
        return pd.DataFrame(rows, columns=["hex", "nbr"])
    hexes = hexes.astype(np.int64)
    return pd.DataFrame({"hex": np.repeat(hexes, 7),
                         "nbr": (hexes[:, None] + _FALLBACK_NBR_OFFSETS[None, :]).ravel()})


def hex_center(hexes):
    """(lat, lon) arrays of hexagon centres."""
    hexes = np.asarray(hexes)
    if len(hexes) and isinstance(hexes[0], str):
        ll = np.array([h3.cell_to_latlng(h) for h in hexes])
        return ll[:, 0], ll[:, 1]
    cells = hexes.astype(np.int64)
    q, r = cells // 10000 - 5000, cells % 10000 - 5000
    x, y = _SIDE * (np.sqrt(3) * q + np.sqrt(3) / 2 * r), _SIDE * 1.5 * r
    return _LAT0 + y / _KM_LAT, _LON0 + x / _KM_LON
