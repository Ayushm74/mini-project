from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from ee_analysis import (
    EarthEngineConfigurationError,
    diagnose_earth_engine,
    run_analysis,
)


def _earth_engine_error_detail(message: str) -> dict[str, str]:
    pid = settings.gee_project_id
    api_lib = "https://console.cloud.google.com/apis/library/earthengine.googleapis.com"
    return {
        "code": "EARTH_ENGINE_CONFIG",
        "message": message,
        "project_id": pid,
        "enable_earth_engine_api": f"{api_lib}?project={pid}" if pid else api_lib,
        "google_cloud_console": (
            f"https://console.cloud.google.com/home/dashboard?project={pid}"
            if pid
            else "https://console.cloud.google.com/"
        ),
        "earth_engine_code_editor": "https://code.earthengine.google.com/",
    }

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


@app.get("/diagnostics/earth-engine")
def earth_engine_diagnostics() -> dict[str, Any]:
    """Check whether Earth Engine can initialize (same project as /analyze)."""
    if not settings.gee_project_id:
        return {
            "ok": False,
            "project_id": None,
            "message": "GEE_PROJECT_ID is not set in backend/.env",
        }
    result = diagnose_earth_engine(settings.gee_project_id)
    if result.get("ok"):
        return result
    return {
        **result,
        **_earth_engine_error_detail(result.get("message") or "Earth Engine failed to initialize"),
    }


@app.post("/analyze")
def analyze(body: AnalyzeRequest) -> dict[str, Any]:
    if not settings.gee_project_id:
        raise HTTPException(
            status_code=503,
            detail=_earth_engine_error_detail(
                "Earth Engine is not configured. Set GEE_PROJECT_ID in backend/.env.",
            ),
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
        raise HTTPException(
            status_code=503,
            detail=_earth_engine_error_detail(str(e)),
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e!s}") from e


@app.post("/export/geojson")
def export_geojson(body: AnalyzeRequest) -> dict[str, Any]:
    """Re-run analysis and return GeoJSON for deforestation vectors (may be heavy)."""
    if not settings.gee_project_id:
        raise HTTPException(
            status_code=503,
            detail=_earth_engine_error_detail("GEE_PROJECT_ID is not configured in backend/.env."),
        )
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
        raise HTTPException(
            status_code=503,
            detail=_earth_engine_error_detail(str(e)),
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
