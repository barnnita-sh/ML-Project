"""Driving distance between hexagons (the fallback method in the team document):
road km = straight-line km between hexagon centres x a detour factor measured on our own trips.

Stored as one small file (artifacts/drive.npz): every pair of hexagons whose road distance is
at most MAX_ROAD_KM. Minutes are NOT stored: the API divides km by the recent taxi speed.
Works with real H3 IDs and with the stand-in grid (IDs are kept as strings).
"""
import numpy as np
from scipy.spatial import cKDTree

from .geo import haversine_km, hex_center

_LAT0 = 13.75
_KM_LAT, _KM_LON = 110.574, 111.320 * np.cos(np.radians(_LAT0))


def detour_factor(trips) -> float:
    """Median of (km driven on a trip) / (straight-line km pickup to drop-off).
    `trips` = pipeline_output/trips tables (columns lat_start, lon_start, lat_end, lon_end, km, is_fake)."""
    t = trips[~trips["is_fake"].to_numpy(bool)]
    straight = haversine_km(t["lat_start"], t["lon_start"], t["lat_end"], t["lon_end"])
    km = t["km"].to_numpy(float)
    ok = (straight >= 1.5) & (km > 0) & (km < 60)
    return float(np.median(km[ok] / straight[ok]))


def build_drive_pairs(hexes, detour: float, max_road_km: float = 10.0):
    """Returns (hex_ids as str, o, d, km): index pairs with road distance <= max_road_km
    (including each hexagon with itself at 0 km)."""
    hexes = np.asarray(hexes)
    ids = np.array([str(h) for h in hexes])
    lat, lon = hex_center(hexes)
    xy = np.c_[(np.asarray(lon) - 100.5) * _KM_LON, (np.asarray(lat) - _LAT0) * _KM_LAT]
    tree = cKDTree(xy)
    r = max_road_km / detour
    coo = tree.sparse_distance_matrix(tree, r, output_type="coo_matrix")
    o, d, v = coo.row, coo.col, coo.data
    keep = o != d
    o, d, km = o[keep], d[keep], v[keep] * detour
    ok = km <= max_road_km
    o, d, km = o[ok], d[ok], km[ok]
    self_ix = np.arange(len(ids))
    o = np.r_[o, self_ix]
    d = np.r_[d, self_ix]
    km = np.r_[km, np.zeros(len(ids))]
    return ids, o.astype(np.int32), d.astype(np.int32), km.astype(np.float32)


def save_drive(path, ids, o, d, km, detour, max_road_km):
    np.savez_compressed(path, hexes=ids, o=o, d=d, km=km, detour=detour, max_road_km=max_road_km)


class DriveTable:
    def __init__(self, hexes, o, d, km, detour, max_road_km):
        order = np.argsort(o, kind="stable")
        self.hexes, self.d, self.km = hexes, d[order], km[order]
        self.start = np.searchsorted(o[order], np.arange(len(hexes) + 1))
        self.index = {h: i for i, h in enumerate(hexes)}
        self.detour, self.max_road_km = float(detour), float(max_road_km)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["hexes"], z["o"], z["d"], z["km"], z["detour"], z["max_road_km"])

    def neighbours(self, hex_id):
        """(destination hexagon IDs, road km) for a hexagon, or None if the hexagon is unknown."""
        i = self.index.get(str(hex_id))
        if i is None:
            return None
        a, b = self.start[i], self.start[i + 1]
        return self.hexes[self.d[a:b]], self.km[a:b]
