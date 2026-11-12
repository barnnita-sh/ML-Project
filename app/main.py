"""Recommendation service: POST /recommend, GET /health, GET /demo/moments, and the demo page at /.

Run locally:   uvicorn app.main:app --reload          (needs the artifacts/ folder, see docs/api_and_deploy.md)
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field, model_validator  # noqa: E402

from taxi.recommend import Recommender  # noqa: E402

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))
STATIC = Path(__file__).resolve().parent / "static"
state = {}


@asynccontextmanager
async def lifespan(app):
    state["rec"] = Recommender.load(ARTIFACTS)      # models, features and drive table are loaded once
    yield


app = FastAPI(title="Bangkok Taxi Area Recommender", version="1.0", lifespan=lifespan)


class RecommendRequest(BaseModel):
    time: str = Field(description="Bangkok local time, e.g. 2025-11-05T14:20")
    minutes_empty: float = Field(ge=0, le=600, description="minutes since the last drop-off")
    hex: Optional[str] = Field(None, description="the driver's H3 hexagon (resolution 8)")
    lat: Optional[float] = None
    lon: Optional[float] = None
    top_k: int = Field(3, ge=1, le=10)

    @model_validator(mode="after")
    def _where(self):
        if self.hex is None and (self.lat is None or self.lon is None):
            raise ValueError("give either hex, or lat and lon")
        return self


@app.get("/health")
def health():
    m = state["rec"].manifest
    return {"status": "ok", "model_version": m.get("model_version"), "rules_version": m.get("rules_version"),
            "training_cutoff": m.get("training_cutoff"), "demo_day": m.get("demo_day"), "built_at": m.get("built_at")}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        return state["rec"].recommend(time=req.time, minutes_empty=req.minutes_empty, hex=req.hex,
                                      lat=req.lat, lon=req.lon, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/demo/moments")
def demo_moments(n: int = 30, seed: int = 0):
    """Sample decision moments from the demo day with what really happened next (no vehicle IDs)."""
    return state["rec"].demo_moments(min(max(n, 1), 200), seed)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
