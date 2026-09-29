"""
OreSense AI – Ingestion Pipeline
==================================
Orchestrates all data adapters and writes observations to PostGIS.
Called by the APScheduler on each source's revisit schedule.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import structlog

from adapters import (
    Sentinel2Adapter,
    Sentinel1Adapter,
    WeatherAdapter,
    AMTAdapter,
    ANTAdapter,
    HyperspectralAdapter,
)
from core.adapter import DataMode
from core.database_local import bulk_insert_observations
from core.grid import cell_id
from config import settings

logger = structlog.get_logger(__name__)

# Adapter singletons
_mode = DataMode.LIVE if settings.is_live else DataMode.SIMULATED
_s2 = Sentinel2Adapter(mode=_mode)
_s1 = Sentinel1Adapter(mode=_mode)
_wx = WeatherAdapter(mode=DataMode.LIVE)   # always try LIVE (free, no auth)
_amt = AMTAdapter()
_ant = ANTAdapter()
_hyp = HyperspectralAdapter()


async def ingest_all(injected_rain_mm: float = 0.0) -> Dict:
    """
    Run all adapters and store per-cell observations.
    Returns a summary dict for the pipeline status endpoint.
    """
    now = datetime.now(timezone.utc)
    rows, cols = settings.grid_rows, settings.grid_cols
    results = {}

    # ── Fetch from each adapter in parallel ───────────────────────────────────
    (s2_obs, s1_obs, wx_obs, amt_obs, ant_obs, hyp_obs) = await asyncio.gather(
        _s2.fetch(),
        _s1.fetch(),
        _wx.fetch(injected_rain_mm=injected_rain_mm),
        _amt.fetch(),
        _ant.fetch(),
        _hyp.fetch(),
        return_exceptions=False,
    )

    results["sentinel2"] = {"mode": s2_obs.mode.value, "ok": s2_obs.ok}
    results["sentinel1"] = {"mode": s1_obs.mode.value, "ok": s1_obs.ok}
    results["weather"] = {"mode": wx_obs.mode.value, "ok": wx_obs.ok}
    results["amt"] = {"mode": amt_obs.mode.value, "ok": amt_obs.ok}
    results["ant"] = {"mode": ant_obs.mode.value, "ok": ant_obs.ok}
    results["hyperspectral"] = {"mode": hyp_obs.mode.value, "ok": hyp_obs.ok}

    # ── Build per-cell records ─────────────────────────────────────────────────
    records = []
    for r in range(rows):
        for c in range(cols):
            cid = cell_id(r, c, cols)

            # Extract scalar from each grid obs (or use scalar weather value)
            def _val(obs, key: str, default=None):
                if obs.error:
                    return default
                v = obs.data.get(key, default)
                if isinstance(v, list):
                    try:
                        return float(np.array(v).ravel()[r * cols + c])
                    except Exception:
                        return default
                return float(v) if v is not None else default

            rec = {
                "time": now,
                "cell_id": cid,
                "source": "fused",
                "data_mode": _consensus_mode(s2_obs, s1_obs, wx_obs, amt_obs),
                "ndvi": _val(s2_obs, "ndvi"),
                "ndvi_anomaly": _val(s2_obs, "ndvi_anomaly"),
                "sar_backscatter": _val(s1_obs, "sar_backscatter"),
                "soil_moisture": _val(s1_obs, "soil_moisture"),
                "lst_anomaly": wx_obs.data.get("lst_anomaly"),
                "hyperspectral_mn_idx": _val(hyp_obs, "hyperspectral_mn_idx"),
                "rainfall_mm": wx_obs.data.get("rainfall_mm"),
                "amt_conductivity": _val(amt_obs, "amt_conductivity"),
                "ant_velocity_pct": _val(ant_obs, "ant_velocity_pct"),
                "insar_deform_mm": _val(s1_obs, "insar_deform_mm"),
            }
            records.append(rec)

    # Write to DB
    try:
        await bulk_insert_observations(records)
        logger.info("ingestion.done", n_cells=len(records), time=now.isoformat())
    except Exception as e:
        logger.error("ingestion.db_error", error=str(e))
        results["db_error"] = str(e)

    results["timestamp"] = now.isoformat()
    results["n_cells"] = len(records)
    return results


def _consensus_mode(*obs) -> str:
    modes = [o.mode.value for o in obs]
    if all(m == "LIVE" for m in modes):
        return "LIVE"
    if any(m == "SIMULATED" for m in modes):
        return "SIMULATED"
    return "CACHED"


async def get_latest_feature_grid() -> Optional[Dict[str, np.ndarray]]:
    """
    Query the DB for the latest observation per cell and return a feature grid.
    Used as input to the reserve model.
    """
    try:
        from core.database_local import get_db_conn

        conn = await get_db_conn()
        cursor = await conn.execute(
            """
            SELECT cell_id, ndvi_anomaly, soil_moisture, insar_deform_mm,
                   hyperspectral_mn_idx, amt_conductivity, ant_velocity_pct
            FROM rs_observations
            WHERE id IN (
                SELECT MAX(id) FROM rs_observations GROUP BY cell_id
            )
            ORDER BY cell_id
            """
        )
        rows_db = await cursor.fetchall()
        await conn.close()

        if not rows_db:
            return None

        R, C = settings.grid_rows, settings.grid_cols
        grid = {
            "ndvi_anomaly": np.zeros((R, C)),
            "soil_moisture": np.full((R, C), 0.25),
            "insar_deform_mm": np.zeros((R, C)),
            "hyperspectral_mn_idx": np.full((R, C), 0.3),
            "amt_conductivity": np.full((R, C), 5.0),
            "ant_velocity_pct": np.zeros((R, C)),
        }

        for row in rows_db:
            idx = row["cell_id"] - 1  # DB uses 1-based
            r = idx // C
            c = idx % C
            if 0 <= r < R and 0 <= c < C:
                for key in grid:
                    v = row[key]
                    if v is not None:
                        grid[key][r, c] = float(v)

        # Add elevation from synthetic grid
        from core.grid import build_grid_gdf
        gdf = build_grid_gdf(**settings.aoi_bounds, rows=R, cols=C)
        elev = np.zeros((R, C))
        for _, row in gdf.iterrows():
            elev[row.row_idx, row.col_idx] = row.elevation_m
        grid["elevation_m"] = elev

        return grid
    except Exception as e:
        logger.error("feature_grid.error", error=str(e))
        return None
