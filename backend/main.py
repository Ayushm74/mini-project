from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from ee_analysis import EarthEngineConfigurationError, run_analysis

app = FastAPI(title="ForestWatch AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(15, ge=5, le=50)
    start_date: str = Field(..., description="ISO date YYYY-MM-DD")
    end_date: str = Field(..., description="ISO date YYYY-MM-DD")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "forestwatch-ai"}


@app.post("/analyze")
def analyze(body: AnalyzeRequest) -> dict[str, Any]:
    if not settings.gee_project_id:
        raise HTTPException(
            status_code=503,
            detail="Earth Engine is not configured. Set GEE_PROJECT_ID in backend/.env",
        )
    try:
        return run_analysis(
            project_id=settings.gee_project_id,
            latitude=body.latitude,
            longitude=body.longitude,
            radius_km=body.radius_km,
            start_date=body.start_date,
            end_date=body.end_date,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except EarthEngineConfigurationError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e!s}") from e


@app.post("/export/geojson")
def export_geojson(body: AnalyzeRequest) -> dict[str, Any]:
    """Re-run analysis and return GeoJSON for deforestation vectors (may be heavy)."""
    if not settings.gee_project_id:
        raise HTTPException(status_code=503, detail="GEE_PROJECT_ID not configured")
    try:
        result = run_analysis(
            project_id=settings.gee_project_id,
            latitude=body.latitude,
            longitude=body.longitude,
            radius_km=body.radius_km,
            start_date=body.start_date,
            end_date=body.end_date,
        )
        gj = result.get("hotspots_geojson")
        if not gj:
            return {"type": "FeatureCollection", "features": []}
        return gj
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except EarthEngineConfigurationError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
