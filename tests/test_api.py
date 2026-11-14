"""Smoke test of the web service with made-up artifacts. Needs fastapi, httpx and pyarrow (CI installs them).
Run:  pytest tests/test_api.py -q
"""
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("pyarrow")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_artifacts import DAY, make_recommender  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    folder = tmp_path_factory.mktemp("artifacts")
    _, hexes = make_recommender(folder=folder)
    os.environ["ARTIFACTS_DIR"] = str(folder)
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        c.hex = str(hexes[len(hexes) // 2])
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_recommend_ok(client):
    r = client.post("/recommend", json={"hex": client.hex, "time": f"{DAY}T14:20", "minutes_empty": 22})
    assert r.status_code == 200
    recs = r.json()["recommendations"]
    assert len(recs) == 3 and [x["rank"] for x in recs] == [1, 2, 3]


def test_recommend_bad_input(client):
    assert client.post("/recommend", json={"time": f"{DAY}T14:20", "minutes_empty": 22}).status_code == 422
    assert client.post("/recommend", json={"hex": "zzz", "time": f"{DAY}T14:20", "minutes_empty": 22}).status_code == 422
    assert client.post("/recommend", json={"hex": client.hex, "time": "2030-01-01T10:00", "minutes_empty": 5}).status_code == 422


def test_demo_moments_and_page(client):
    m = client.get("/demo/moments?n=5").json()
    assert len(m) == 5 and "vid" not in m[0]
    assert client.get("/").status_code == 200
