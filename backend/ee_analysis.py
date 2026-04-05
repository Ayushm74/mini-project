"""
ForestWatch AI — Google Earth Engine analysis (Landsat 8/9 + Sentinel-2).
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any

import ee


NDVI_LOSS_THRESHOLD = -0.15


class EarthEngineConfigurationError(RuntimeError):
    """Raised when Earth Engine is missing required project or auth setup."""


def _is_ee_api_or_project_misconfiguration(exc: Exception) -> bool:
    """True when enabling the API / fixing the Cloud project is the fix (not re-auth)."""
    t = str(exc).lower()
    needles = (
        "has not been used in project",
        "is disabled",
        "has not been enabled",
        "api has not been enabled",
        "service_disabled",
        "earthengine.googleapis.com",
        "permission_denied",
        "billing has not been enabled",
        "access not configured",
        "earth engine api",
    )
    return any(n in t for n in needles)


def _looks_like_missing_credentials(exc: Exception) -> bool:
    t = str(exc).lower()
    return any(
        x in t
        for x in (
            "could not find default credentials",
            "reauthentication is needed",
            "not authenticated",
            "application default credentials",
            "refresh token",
            "unable to authenticate",
        )
    )


def _normalize_ee_error(project_id: str, exc: Exception) -> EarthEngineConfigurationError:
    message = str(exc)
    lowered = message.lower()

    if _is_ee_api_or_project_misconfiguration(exc):
        return EarthEngineConfigurationError(
            "Google Cloud returned an error for this project—usually the Earth Engine API is not "
            "enabled yet, billing is off, or the account does not have access. "
            f"Project: {project_id}. Raw error: {message}"
        )

    if "google earth engine not initialized" in lowered or (
        "credentials" in lowered and not _is_ee_api_or_project_misconfiguration(exc)
    ):
        return EarthEngineConfigurationError(
            "Earth Engine credentials are missing or invalid. Run `earthengine authenticate` in a "
            "terminal (or set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON with EE "
            "access), then restart the backend. "
            f"Raw error: {message}"
        )

    return EarthEngineConfigurationError(
        f"Earth Engine initialization failed for project '{project_id}': {message}"
    )


def _init_ee(project_id: str) -> None:
    if not project_id:
        raise EarthEngineConfigurationError(
            "Earth Engine is not configured. Set GEE_PROJECT_ID in backend/.env."
        )
    try:
        ee.Initialize(project=project_id)
    except Exception as first_error:
        if _is_ee_api_or_project_misconfiguration(first_error):
            raise _normalize_ee_error(project_id, first_error) from first_error
        if not _looks_like_missing_credentials(first_error):
            raise _normalize_ee_error(project_id, first_error) from first_error
        try:
            ee.Authenticate()
            ee.Initialize(project=project_id)
        except Exception as second_error:
            raise _normalize_ee_error(project_id, second_error) from second_error


def diagnose_earth_engine(project_id: str) -> dict[str, Any]:
    """Lightweight check for /diagnostics (does not run analysis)."""
    out: dict[str, Any] = {
        "project_id": project_id or None,
        "ok": False,
        "raw_error": None,
    }
    if not project_id:
        out["hint"] = "Set GEE_PROJECT_ID in backend/.env"
        return out
    try:
        ee.Initialize(project=project_id)
        out["ok"] = True
        out["message"] = "Earth Engine initialized successfully for this project."
    except Exception as e:
        out["raw_error"] = str(e)
        out["message"] = str(_normalize_ee_error(project_id, e))
    return out


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s[:10])


def _split_period(start: dt.date, end: dt.date) -> tuple[tuple[dt.date, dt.date], tuple[dt.date, dt.date]]:
    """Split [start, end] into two contiguous halves (before / after)."""
    total_days = (end - start).days
    if total_days < 14:
        raise ValueError("Date range must span at least 14 days.")
    mid = start + dt.timedelta(days=total_days // 2)
    before = (start, mid)
    after = (mid + dt.timedelta(days=1), end)
    return before, after


def _duration_days(start: dt.date, end: dt.date) -> int:
    return (end - start).days


def _use_sentinel(start: dt.date, end: dt.date) -> bool:
    return _duration_days(start, end) <= 366


def _mask_s2_clouds(img: ee.Image) -> ee.Image:
    qa = img.select("QA60")
    cloud = qa.bitwiseAnd(1 << 10).neq(0)
    cirrus = qa.bitwiseAnd(1 << 11).neq(0)
    return img.updateMask(cloud.Or(cirrus).Not())


def _landsat_collection(
    region: ee.Geometry, d0: dt.date, d1: dt.date, cloud_cover: int
) -> ee.ImageCollection:
    start, end = d0.isoformat(), d1.isoformat()
    l8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover))
    )
    l9 = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover))
    )
    return l8.merge(l9)


def _sentinel_collection(region: ee.Geometry, d0: dt.date, d1: dt.date) -> ee.ImageCollection:
    start, end = d0.isoformat(), d1.isoformat()
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
        .map(_mask_s2_clouds)
    )
    return col


def _median_landsat(region: ee.Geometry, d0: dt.date, d1: dt.date) -> ee.Image:
    col = _landsat_collection(region, d0, d1, cloud_cover=25)
    img = col.median().clip(region)
    return img


def _median_sentinel(region: ee.Geometry, d0: dt.date, d1: dt.date) -> ee.Image:
    col = _sentinel_collection(region, d0, d1)
    img = col.median().clip(region)
    return img


def _calc_ndvi_landsat(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")


def _calc_ndvi_sentinel(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(["B8", "B4"]).rename("NDVI")


def _composite_and_ndvi(
    region: ee.Geometry, d0: dt.date, d1: dt.date, sentinel: bool
) -> ee.Image:
    if sentinel:
        raw = _median_sentinel(region, d0, d1)
        return _calc_ndvi_sentinel(raw)
    raw = _median_landsat(region, d0, d1)
    return _calc_ndvi_landsat(raw)


def _scale_for_sentinel(sentinel: bool) -> int:
    return 10 if sentinel else 30


def _tile_url(map_id: dict[str, Any]) -> str:
    tf = map_id.get("tile_fetcher")
    if tf is not None and hasattr(tf, "url_format"):
        return str(tf.url_format)
    mid = map_id["mapid"]
    token = map_id["token"]
    return f"https://earthengine.googleapis.com/map/{mid}/{{z}}/{{x}}/{{y}}?token={token}"


def _region_geometry(lat: float, lon: float, radius_km: float) -> ee.Geometry:
    point = ee.Geometry.Point([float(lon), float(lat)])
    return point.buffer(float(radius_km) * 1000)


def _reduce_mean_ndvi(ndvi: ee.Image, region: ee.Geometry, scale: int) -> float | None:
    d = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()
    v = d.get("NDVI")
    if v is None:
        return None
    return float(v)


def _yearly_trend(
    region: ee.Geometry,
    start: dt.date,
    end: dt.date,
    sentinel: bool,
) -> list[dict[str, float]]:
    """Client-side yearly medians for trend chart (limited years for latency)."""
    out: list[dict[str, float]] = []
    y0, y1 = start.year, end.year
    scale = _scale_for_sentinel(sentinel)
    for year in range(y0, y1 + 1):
        y_start = dt.date(year, 1, 1)
        y_end = dt.date(year, 12, 31)
        if y_end < start or y_start > end:
            continue
        a = max(y_start, start)
        b = min(y_end, end)
        if (b - a).days < 30:
            continue
        try:
            ndvi = _composite_and_ndvi(region, a, b, sentinel)
            m = _reduce_mean_ndvi(ndvi, region, scale)
            if m is not None and not math.isnan(m):
                out.append({"year": float(year), "mean_ndvi": round(m, 4)})
        except Exception:
            continue
    return out


def _hotspots_from_mask(
    mask: ee.Image, region: ee.Geometry, scale: int, max_points: int = 12
) -> list[list[float]]:
    """Sample points on deforestation mask as hotspot proxies."""
    try:
        sample = mask.selfMask().sample(
            region=region,
            scale=scale,
            numPixels=max(50, max_points * 8),
            geometries=True,
            seed=42,
        ).limit(max_points)
        feats = sample.getInfo().get("features", [])
        coords: list[list[float]] = []
        for f in feats:
            geom = f.get("geometry")
            if not geom or geom["type"] != "Point":
                continue
            c = geom["coordinates"]
            coords.append([float(c[1]), float(c[0])])  # lat, lon
        return coords[:max_points]
    except Exception:
        return []


def _deforestation_geojson(
    loss_mask: ee.Image, region: ee.Geometry, scale: int
) -> dict[str, Any] | None:
    try:
        vecs = loss_mask.selfMask().reduceToVectors(
            geometry=region,
            scale=max(scale * 3, 30),
            maxPixels=1e9,
            geometryType="polygon",
            eightConnected=False,
            maxFeatures=200,
        )
        info = vecs.getInfo()
        return info
    except Exception:
        return None


def _build_insights(
    loss_pct: float,
    loss_km2: float,
    mean_change: float | None,
    duration_label: str,
) -> dict[str, Any]:
    summary_parts = [
        f"Over {duration_label}, the selected area shows approximately {loss_pct:.1f}% vegetation loss "
        f"across roughly {loss_km2:.2f} km² of declining NDVI (threshold {NDVI_LOSS_THRESHOLD})."
    ]
    if mean_change is not None:
        summary_parts.append(f"Mean NDVI change is {mean_change:+.3f}.")

    causes: list[str] = []
    if loss_pct > 15:
        causes.extend(
            [
                "Strong negative NDVI trend consistent with clearing or major canopy disturbance.",
                "Possible drivers include urban expansion, agriculture, or timber extraction.",
            ]
        )
    elif loss_pct > 5:
        causes.append("Moderate vegetation decline may reflect seasonal stress, fire, or gradual land-use change.")
    else:
        causes.append("Vegetation appears relatively stable; residual change may include seasonality and noise.")

    if loss_pct > 8:
        causes.append("Climate variability (drought, heatwaves) can amplify browning in already stressed forests.")

    actions = [
        "Validate findings with high-resolution imagery and field visits before enforcement actions.",
        "Prioritize riparian buffers and connectivity corridors in areas with concentrated loss.",
        "Engage local communities and land registries to distinguish legal land use from illegal activity.",
        "Establish repeat monitoring (e.g., quarterly Sentinel-2) for active alerts.",
    ]
    return {"summary": " ".join(summary_parts), "possible_causes": causes, "suggested_actions": actions}


def _eco_score(mean_after: float | None, loss_pct: float) -> int:
    base = 50
    if mean_after is not None:
        base += int(max(-30, min(40, (mean_after - 0.2) * 120)))
    base -= int(min(45, loss_pct * 2))
    return int(max(0, min(100, base)))


def run_analysis(
    *,
    project_id: str,
    latitude: float,
    longitude: float,
    radius_km: float,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    _init_ee(project_id)

    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end <= start:
        raise ValueError("end_date must be after start_date.")

    (b0, b1), (a0, a1) = _split_period(start, end)
    sentinel = _use_sentinel(start, end)
    scale = _scale_for_sentinel(sentinel)

    region = _region_geometry(latitude, longitude, radius_km)

    ndvi_before = _composite_and_ndvi(region, b0, b1, sentinel)
    ndvi_after = _composite_and_ndvi(region, a0, a1, sentinel)
    ndvi_change = ndvi_after.subtract(ndvi_before)

    loss_mask = ndvi_change.lt(NDVI_LOSS_THRESHOLD)
    loss_layer = ndvi_change.updateMask(loss_mask)

    vis_ndvi = {"bands": ["NDVI"], "min": 0, "max": 0.85, "palette": ["#0f172a", "#14532d", "#22c55e", "#86efac"]}
    vis_loss = {
        "bands": ["NDVI"],
        "min": -0.55,
        "max": -0.08,
        "palette": ["#7f1d1d", "#dc2626", "#fb923c", "#fcd34d"],
    }
    vis_change = {"bands": ["NDVI"], "min": -0.4, "max": 0.4, "palette": ["#b91c1c", "#78716c", "#22c55e"]}

    before_id = ndvi_before.getMapId(vis_ndvi)
    after_id = ndvi_after.getMapId(vis_ndvi)
    change_id = ndvi_change.getMapId(vis_change)
    heat_id = loss_layer.getMapId(vis_loss)

    pixel_area = ee.Image.pixelArea()
    loss_area_img = loss_mask.rename("loss").multiply(pixel_area)
    veg_before = ndvi_before.gt(0.2)
    veg_area_img = veg_before.rename("veg").multiply(pixel_area)

    stats = loss_area_img.addBands(veg_area_img).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()

    loss_m2 = float(stats.get("loss") or 0)
    veg_m2 = float(stats.get("veg") or 0)
    loss_km2 = loss_m2 / 1e6
    loss_pct = (loss_m2 / veg_m2 * 100.0) if veg_m2 > 0 else 0.0

    change_mean = ndvi_change.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()
    mean_change = change_mean.get("NDVI")
    if mean_change is not None:
        mean_change = float(mean_change)

    mean_after = _reduce_mean_ndvi(ndvi_after, region, scale)

    years_span = (end - start).days / 365.25
    deforestation_rate = loss_km2 / years_span if years_span > 0 else loss_km2

    hotspots = _hotspots_from_mask(loss_mask, region, scale)
    fc_geojson = _deforestation_geojson(loss_mask, region, scale)

    trend = _yearly_trend(region, start, end, sentinel)

    total_days = _duration_days(start, end)
    duration_label = f"{total_days} days"
    if total_days >= 300:
        duration_label = f"~{total_days / 365.25:.1f} years"

    warning = None
    if total_days <= 200:
        warning = (
            "Short-term analysis may be affected by seasonal vegetation changes. "
            "Prefer ≥12-month windows or interpret alongside phenology."
        )

    insights = _build_insights(loss_pct, loss_km2, mean_change, duration_label)

    def _thumb(img: ee.Image, vis: dict[str, Any]) -> str:
        pals = vis.get("palette") or []
        pal = ",".join(str(p).replace("#", "") for p in pals)
        return img.getThumbURL(
            {
                "min": vis["min"],
                "max": vis["max"],
                "bands": vis["bands"],
                "palette": pal,
                "region": region,
                "dimensions": 1024,
                "format": "png",
            }
        )

    exports_png = {
        "ndvi_before": _thumb(ndvi_before, vis_ndvi),
        "ndvi_after": _thumb(ndvi_after, vis_ndvi),
        "ndvi_change": _thumb(ndvi_change, vis_change),
        "deforestation": _thumb(loss_layer, vis_loss),
    }

    return {
        "center": {"lat": latitude, "lon": longitude},
        "radius_km": radius_km,
        "bounds": region.bounds().getInfo(),
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "before_period": {"start": b0.isoformat(), "end": b1.isoformat()},
        "after_period": {"start": a0.isoformat(), "end": a1.isoformat()},
        "satellite": "sentinel2" if sentinel else "landsat89",
        "scale_m": scale,
        "layers": {
            "ndvi_before": {"url": _tile_url(before_id), "opacity": 0.9},
            "ndvi_after": {"url": _tile_url(after_id), "opacity": 0.9},
            "ndvi_change": {"url": _tile_url(change_id), "opacity": 0.85},
            "deforestation": {"url": _tile_url(heat_id), "opacity": 0.9},
        },
        "analytics": {
            "affected_area_km2": round(loss_km2, 3),
            "vegetation_loss_percent": round(loss_pct, 2),
            "hotspot_count": len(hotspots),
            "top_hotspots": hotspots,
            "mean_ndvi_change": round(mean_change, 4) if mean_change is not None else None,
            "mean_ndvi_after": round(mean_after, 4) if mean_after is not None else None,
            "deforestation_rate_km2_per_year": round(deforestation_rate, 4),
            "years_span": round(years_span, 2),
        },
        "trend": trend,
        "insights": insights,
        "eco_score": _eco_score(mean_after, loss_pct),
        "warning": warning,
        "hotspots_geojson": fc_geojson,
        "threshold_ndvi_change": NDVI_LOSS_THRESHOLD,
        "exports": {"png": exports_png},
    }
