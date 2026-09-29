"""
OreSense AI – FastAPI Application (Local Mode)
================================================
REST API + WebSocket + APScheduler pipeline orchestration.
Modified for local development without Docker/PostgreSQL.

Routes:
  GET  /api/grid              → current reserve map
  GET  /api/sector/{id}       → single cell detail
  POST /api/predict/sector    → on-demand cell prediction
  GET  /api/shortfall         → current risk, drivers, forecast
  GET  /api/actions           → top 3 RL-recommended actions
  GET  /api/causal            → causal effects and refutation
  GET  /api/federated/status  → FL round metrics
  POST /api/simulate/event    → inject simulator event
  POST /api/groundtruth       → post ground-truth drilling result
  GET  /api/sources/status    → data source health
  WS   /ws                    → live push of new predictions

Scheduler:
  Every 15 min:  weather + full ingestion
  Every 2 min:   reserve model re-inference
  Every 30 min:  shortfall prediction refresh
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add backend dir to path (works on both Docker and local)
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from config import settings
from pipeline.ingestion import ingest_all, get_latest_feature_grid

logger = structlog.get_logger(__name__)

# ── Import DB helpers (SQLite local mode) ──────────────────────────────────────
from core.database_local import get_db_conn

# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, data: Dict):
        msg = json.dumps(data, default=str)
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self.active -= dead


manager = ConnectionManager()

# ── In-memory state cache (updated by scheduler) ──────────────────────────────
_state: Dict[str, Any] = {
    "grid": None,
    "shortfall": None,
    "actions": None,
    "causal": None,
    "sources": {},
    "pipeline_status": {},
    "last_update": None,
}


# ── Scheduler tasks ───────────────────────────────────────────────────────────
async def run_ingestion_and_inference():
    """Full pipeline: ingest → reserve model → shortfall → broadcast."""
    try:
        t0 = datetime.now(timezone.utc)
        logger.info("pipeline.start")

        # 1. Get injected event parameters
        injected_rain = await _get_injected_rain()

        # 2. Ingest all sensors
        src_status = await ingest_all(injected_rain_mm=injected_rain)
        _state["sources"] = src_status

        # 3. Reserve model inference
        await run_reserve_inference()

        # 4. Shortfall prediction
        await run_shortfall_inference(injected_rain=injected_rain)

        # 5. Actions (RL twin)
        await run_actions_inference()

        latency_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
        _state["pipeline_status"] = {
            "last_run": t0.isoformat(),
            "latency_ms": latency_ms,
            "stages": {
                "ingestion": "OK",
                "reserve": "OK",
                "shortfall": "OK",
                "actions": "OK",
            },
        }
        _state["last_update"] = datetime.now(timezone.utc).isoformat()

        # Broadcast to WebSocket clients
        await manager.broadcast({
            "type": "update",
            "grid": _state["grid"],
            "shortfall": _state["shortfall"],
            "actions": _state["actions"],
            "timestamp": _state["last_update"],
        })
        logger.info("pipeline.done", latency_ms=latency_ms)
    except Exception as e:
        logger.error("pipeline.error", error=str(e))
        _state["pipeline_status"]["last_error"] = str(e)


async def run_reserve_inference():
    try:
        from models.reserve_model import predict_grid, classify_reserve, get_version_info
        from core.database_local import upsert_reserve_estimates
        from core.grid import cell_id

        grid = await get_latest_feature_grid()
        if grid is None:
            return

        preds = predict_grid(grid)
        ore_prob = preds["ore_prob"]
        ore_std = preds["ore_uncertainty"]
        R, C = settings.grid_rows, settings.grid_cols
        version_info = get_version_info()

        records = []
        grid_data = []
        for r in range(R):
            for c in range(C):
                prob = float(ore_prob[r, c])
                std = float(ore_std[r, c])
                cls = classify_reserve(prob)
                cid = cell_id(r, c, C)
                records.append({
                    "time": datetime.now(timezone.utc),
                    "cell_id": cid,
                    "model_version": version_info.get("version", "unknown"),
                    "ore_prob": prob,
                    "ore_uncertainty": std,
                    "reserve_class": cls,
                    "feature_json": {k: float(grid[k][r, c]) for k in grid},
                })
                grid_data.append({
                    "cell_id": cid,
                    "row": r,
                    "col": c,
                    "ore_prob": round(prob, 4),
                    "ore_uncertainty": round(std, 4),
                    "reserve_class": cls,
                })

        await upsert_reserve_estimates(records)
        _state["grid"] = grid_data
    except Exception as e:
        logger.error("reserve_inference.error", error=str(e))


async def run_shortfall_inference(injected_rain: float = 0.0):
    try:
        from models.shortfall_model import predict_current
        from core.database_local import insert_shortfall_prediction

        # Get injected events
        breakdown_h = await _get_injected_breakdown_h()
        crew_shortage = await _get_injected_crew_shortage()

        result = predict_current(
            injected_rain=injected_rain,
            injected_breakdown_h=breakdown_h,
            injected_crew_shortage=crew_shortage,
        )

        await insert_shortfall_prediction(
            risk_pct=result["risk_pct"],
            uncertainty=result["uncertainty"],
            top_drivers=result["top_drivers"],
            forecast_json=result["forecast"],
            model_version=result["version"],
        )
        _state["shortfall"] = result
    except Exception as e:
        logger.error("shortfall_inference.error", error=str(e))


async def run_actions_inference():
    try:
        from models.rl_twin import get_top_actions
        shortfall = _state.get("shortfall") or {}
        risk = shortfall.get("risk_pct", 20.0)
        actions = get_top_actions(current_shortfall=risk, rainfall_mm=await _get_injected_rain())
        _state["actions"] = actions
    except Exception as e:
        logger.error("actions_inference.error", error=str(e))


async def _get_injected_rain() -> float:
    try:
        conn = await get_db_conn()
        cursor = await conn.execute(
            "SELECT parameters FROM simulator_events WHERE event_type='heavy_rain' AND active_until > datetime('now') ORDER BY created_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        await conn.close()
        if row:
            return float(json.loads(row["parameters"]).get("rain_mm", 0))
    except Exception:
        pass
    return 0.0


async def _get_injected_breakdown_h() -> float:
    try:
        conn = await get_db_conn()
        cursor = await conn.execute(
            "SELECT parameters FROM simulator_events WHERE event_type='equipment_breakdown' AND active_until > datetime('now') ORDER BY created_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        await conn.close()
        if row:
            return float(json.loads(row["parameters"]).get("hours", 6))
    except Exception:
        pass
    return 0.0


async def _get_injected_crew_shortage() -> int:
    try:
        conn = await get_db_conn()
        cursor = await conn.execute(
            "SELECT parameters FROM simulator_events WHERE event_type='crew_shortage' AND active_until > datetime('now') ORDER BY created_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        await conn.close()
        if row:
            return int(json.loads(row["parameters"]).get("shortage", 30))
    except Exception:
        pass
    return 0


# ── App lifecycle ─────────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler(timezone="UTC")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load causal results from DB
    try:
        conn = await get_db_conn()
        cursor = await conn.execute(
            "SELECT results_json FROM causal_estimates ORDER BY computed_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        await conn.close()
        if row:
            _state["causal"] = json.loads(row["results_json"])
    except Exception:
        pass

    # Warm up grid
    await run_ingestion_and_inference()

    # Schedule recurring tasks
    scheduler.add_job(run_ingestion_and_inference, "interval", minutes=15, id="pipeline")
    scheduler.add_job(run_reserve_inference, "interval", minutes=2, id="reserve")
    scheduler.add_job(run_shortfall_inference, "interval", minutes=30, id="shortfall")
    scheduler.start()
    logger.info("scheduler.started")
    yield
    scheduler.shutdown()


app = FastAPI(
    title="OreSense AI",
    description="AI/ML Platform for MOIL Manganese Reserve Mapping & Shortfall Prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Routes ───────────────────────────────────────────────────────────────
@app.get("/api/grid")
async def get_grid():
    """Current reserve map: per-cell ore_prob, class, uncertainty."""
    if _state["grid"] is None:
        await run_reserve_inference()
    grid_data = _state["grid"] or []
    # Augment with lat/lon from DB
    try:
        conn = await get_db_conn()
        cursor = await conn.execute(
            "SELECT id, row_idx, col_idx, centroid_lat AS lat, centroid_lon AS lon, elevation_m FROM grid_cells"
        )
        rows = await cursor.fetchall()
        await conn.close()
        coord_map = {r["id"]: {"lat": r["lat"], "lon": r["lon"], "elevation_m": r["elevation_m"]} for r in rows}
        for cell in grid_data:
            coords = coord_map.get(cell["cell_id"], {})
            cell.update(coords)
    except Exception:
        pass
    return {"cells": grid_data, "last_update": _state.get("last_update")}


@app.get("/api/sector/{cell_id}")
async def get_sector(cell_id: int):
    """Detailed feature breakdown for a single grid cell."""
    try:
        conn = await get_db_conn()
        # Latest estimate
        cursor = await conn.execute(
            "SELECT * FROM reserve_estimates WHERE cell_id=? ORDER BY time DESC LIMIT 1", (cell_id,)
        )
        est = await cursor.fetchone()
        # Latest observation
        cursor = await conn.execute(
            "SELECT * FROM rs_observations WHERE cell_id=? ORDER BY time DESC LIMIT 1", (cell_id,)
        )
        obs = await cursor.fetchone()
        await conn.close()
        if not est:
            raise HTTPException(status_code=404, detail="Cell not found")
        return {
            "cell_id": cell_id,
            "estimate": dict(est),
            "observation": dict(obs) if obs else {},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SectorPredictRequest(BaseModel):
    features: Dict[str, float]


@app.post("/api/predict/sector")
async def predict_sector(req: SectorPredictRequest):
    """On-demand prediction for a single cell given feature overrides."""
    from models.reserve_model import predict_grid, classify_reserve
    import numpy as np
    feature_grid = {k: np.array([[v]]) for k, v in req.features.items()}
    try:
        preds = predict_grid(feature_grid)
        prob = float(preds["ore_prob"][0, 0])
        std = float(preds["ore_uncertainty"][0, 0])
        return {
            "ore_prob": round(prob, 4),
            "ore_uncertainty": round(std, 4),
            "reserve_class": classify_reserve(prob),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/shortfall")
async def get_shortfall():
    """Current shortfall risk, top causal drivers, 30-day forecast."""
    if _state["shortfall"] is None:
        await run_shortfall_inference()
    return _state["shortfall"] or {"error": "Not yet computed"}


@app.get("/api/actions")
async def get_actions():
    """Top 3 RL-recommended corrective actions."""
    if _state["actions"] is None:
        await run_actions_inference()
    return {"actions": _state["actions"] or [], "data_mode": "SIMULATED"}


@app.get("/api/causal")
async def get_causal():
    """Causal effect estimates and refutation results."""
    if _state["causal"] is None:
        try:
            conn = await get_db_conn()
            cursor = await conn.execute(
                "SELECT results_json FROM causal_estimates ORDER BY computed_at DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            await conn.close()
            if row:
                _state["causal"] = json.loads(row["results_json"])
        except Exception:
            pass
    return _state["causal"] or {"error": "Causal model not yet run"}


@app.get("/api/federated/status")
async def get_federated_status():
    """Federated learning round metrics."""
    try:
        conn = await get_db_conn()
        cursor = await conn.execute(
            "SELECT round_num, completed_at, global_accuracy, client_metrics FROM fl_rounds ORDER BY round_num"
        )
        rows = await cursor.fetchall()
        await conn.close()
        return {
            "rounds": [
                {
                    "round": r["round_num"],
                    "completed_at": r["completed_at"],
                    "global_accuracy": r["global_accuracy"],
                    "clients": json.loads(r["client_metrics"]) if r["client_metrics"] else [],
                }
                for r in rows
            ]
        }
    except Exception as e:
        return {"rounds": [], "error": str(e)}


class SimulateEventRequest(BaseModel):
    event_type: str   # heavy_rain | equipment_breakdown | crew_shortage
    parameters: Dict[str, Any]
    duration_minutes: int = 60


@app.post("/api/simulate/event")
async def simulate_event(req: SimulateEventRequest):
    """Inject a simulator event (heavy rain, breakdown, crew shortage)."""
    valid = {"heavy_rain", "equipment_breakdown", "crew_shortage"}
    if req.event_type not in valid:
        raise HTTPException(status_code=400, detail=f"event_type must be one of {valid}")
    try:
        conn = await get_db_conn()
        active_until = datetime.now(timezone.utc) + timedelta(minutes=req.duration_minutes)
        await conn.execute(
            """
            INSERT INTO simulator_events (event_type, parameters, active_until)
            VALUES (?, ?, ?)
            """,
            (req.event_type, json.dumps(req.parameters), active_until.isoformat()),
        )
        await conn.commit()
        await conn.close()
        # Trigger immediate pipeline re-run
        asyncio.create_task(run_ingestion_and_inference())
        return {"status": "injected", "active_until": active_until.isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class GroundTruthRequest(BaseModel):
    cell_id: int
    actual_grade: float   # 0–1
    notes: str = ""


@app.post("/api/groundtruth")
async def post_groundtruth(req: GroundTruthRequest):
    """
    Post a simulated drilling ground-truth result.
    Triggers model retraining (async).
    """
    # For demo: just trigger a retraining job in background
    asyncio.create_task(_retrain_reserve_model())
    return {
        "status": "received",
        "cell_id": req.cell_id,
        "actual_grade": req.actual_grade,
        "message": "Retraining triggered in background",
    }


async def _retrain_reserve_model():
    try:
        from models.reserve_model import train as train_reserve
        result = train_reserve(rows=settings.grid_rows, cols=settings.grid_cols, n_mines=40)
        logger.info("retrain.done", version=result["version"])
        await run_reserve_inference()
        await manager.broadcast({"type": "model_updated", "version": result["version"]})
    except Exception as e:
        logger.error("retrain.error", error=str(e))


@app.get("/api/sources/status")
async def get_sources_status():
    """Health of all data source adapters."""
    from adapters import (
        Sentinel2Adapter, Sentinel1Adapter, WeatherAdapter,
        AMTAdapter, ANTAdapter, HyperspectralAdapter
    )
    from core.adapter import DataMode
    mode = DataMode.LIVE if settings.is_live else DataMode.SIMULATED
    return {
        "mode": settings.mode,
        "sources": [
            {"name": "Sentinel-2 (NDVI)", "mode": mode.value, "revisit_days": 5, "note": "Surface vegetation/Mn index"},
            {"name": "Sentinel-1 (SAR)", "mode": "SIMULATED", "revisit_days": 12, "note": "InSAR: CACHED (precomputed)"},
            {"name": "Weather (Open-Meteo)", "mode": "LIVE", "revisit_hours": 1, "note": "No auth required"},
            {"name": "AMT/CSAMT", "mode": "SIMULATED", "note": "Requires field survey"},
            {"name": "ANT Tomography", "mode": "SIMULATED", "note": "Requires seismic network"},
            {"name": "Hyperspectral", "mode": "SIMULATED", "note": "No EnMAP scene for AOI"},
            {"name": "IoT Equipment", "mode": "SIMULATED", "note": "8 machines via MQTT"},
        ],
        "pipeline": _state.get("pipeline_status", {}),
    }


@app.get("/api/iot")
async def get_iot():
    """Latest IoT telemetry for all machines."""
    try:
        from core.database_local import get_latest_iot
        machines = await get_latest_iot()
        return {"machines": machines, "data_mode": "SIMULATED"}
    except Exception as e:
        return {"machines": [], "error": str(e)}


@app.get("/api/models")
async def get_model_registry():
    """Model registry with versions and metrics."""
    try:
        conn = await get_db_conn()
        cursor = await conn.execute(
            "SELECT model_name, version, trained_at, metrics_json, is_active FROM model_registry ORDER BY trained_at DESC"
        )
        rows = await cursor.fetchall()
        await conn.close()
        return {
            "models": [
                {
                    "name": r["model_name"],
                    "version": r["version"],
                    "trained_at": r["trained_at"],
                    "metrics": json.loads(r["metrics_json"]) if r["metrics_json"] else {},
                    "is_active": bool(r["is_active"]),
                }
                for r in rows
            ]
        }
    except Exception as e:
        return {"models": [], "error": str(e)}


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send current state immediately on connect
        await websocket.send_text(
            json.dumps(
                {
                    "type": "init",
                    "grid": _state["grid"],
                    "shortfall": _state["shortfall"],
                    "actions": _state["actions"],
                    "timestamp": _state["last_update"],
                },
                default=str,
            )
        )
        # Keep alive
        while True:
            data = await websocket.receive_text()
            # Handle ping
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
