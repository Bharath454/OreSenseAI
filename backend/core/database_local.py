"""
OreSense AI – Local Database helpers (SQLite via aiosqlite)
============================================================
Drop-in replacement for core/database.py when running without PostgreSQL.
Uses a local SQLite file instead of PostgreSQL/PostGIS/TimescaleDB.
"""
from __future__ import annotations

import json
import os
import sqlite3
import aiosqlite
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings

# Database file location
_BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_PATH = _BACKEND_DIR / "data" / "oresense.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS grid_cells (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    row_idx     INTEGER NOT NULL,
    col_idx     INTEGER NOT NULL,
    centroid_lat REAL,
    centroid_lon REAL,
    elevation_m REAL DEFAULT 0,
    UNIQUE (row_idx, col_idx)
);

CREATE TABLE IF NOT EXISTS rs_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    time            TEXT NOT NULL,
    cell_id         INTEGER NOT NULL REFERENCES grid_cells(id),
    source          TEXT NOT NULL,
    data_mode       TEXT NOT NULL,
    ndvi            REAL,
    ndvi_anomaly    REAL,
    sar_backscatter REAL,
    soil_moisture   REAL,
    lst_anomaly     REAL,
    hyperspectral_mn_idx REAL,
    rainfall_mm     REAL,
    amt_conductivity REAL,
    ant_velocity_pct REAL,
    insar_deform_mm  REAL
);

CREATE TABLE IF NOT EXISTS reserve_estimates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    time            TEXT NOT NULL,
    cell_id         INTEGER NOT NULL REFERENCES grid_cells(id),
    model_version   TEXT NOT NULL,
    ore_prob        REAL,
    ore_uncertainty REAL,
    reserve_class   TEXT,
    feature_json    TEXT
);

CREATE TABLE IF NOT EXISTS production_records (
    date            TEXT PRIMARY KEY,
    site            TEXT NOT NULL DEFAULT 'Balaghat',
    planned_tonnes  REAL,
    actual_tonnes   REAL,
    shortfall_pct   REAL,
    rainfall_mm     REAL,
    equipment_downtime_h REAL,
    crew_available  INTEGER,
    blasting_delays_h REAL
);

CREATE TABLE IF NOT EXISTS shortfall_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    time            TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    risk_pct        REAL,
    uncertainty     REAL,
    top_drivers     TEXT,
    forecast_json   TEXT
);

CREATE TABLE IF NOT EXISTS causal_estimates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at     TEXT DEFAULT (datetime('now')),
    model_version   TEXT,
    results_json    TEXT
);

CREATE TABLE IF NOT EXISTS iot_telemetry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    time            TEXT NOT NULL,
    machine_id      TEXT NOT NULL,
    lat             REAL,
    lon             REAL,
    vibration_g     REAL,
    fuel_rate_lph   REAL,
    state           TEXT,
    health_score    REAL
);

CREATE TABLE IF NOT EXISTS model_registry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL,
    version         TEXT NOT NULL,
    trained_at      TEXT DEFAULT (datetime('now')),
    metrics_json    TEXT,
    is_active       INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS fl_rounds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    round_num       INTEGER,
    completed_at    TEXT DEFAULT (datetime('now')),
    global_accuracy REAL,
    client_metrics  TEXT
);

CREATE TABLE IF NOT EXISTS simulator_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT DEFAULT (datetime('now')),
    event_type      TEXT,
    parameters      TEXT,
    active_until    TEXT
);
"""


def init_db_sync():
    """Create all tables synchronously (for startup)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA_SQL)
    conn.close()


async def get_db_conn() -> aiosqlite.Connection:
    """Get an async SQLite connection."""
    conn = await aiosqlite.connect(str(DB_PATH))
    conn.row_factory = aiosqlite.Row
    return conn


async def bulk_insert_observations(records: List[Dict[str, Any]]) -> None:
    """Bulk-insert remote-sensing observations."""
    if not records:
        return
    conn = await get_db_conn()
    try:
        await conn.executemany(
            """
            INSERT INTO rs_observations
            (time, cell_id, source, data_mode,
             ndvi, ndvi_anomaly, sar_backscatter, soil_moisture,
             lst_anomaly, hyperspectral_mn_idx, rainfall_mm,
             amt_conductivity, ant_velocity_pct, insar_deform_mm)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    r["time"].isoformat() if isinstance(r["time"], datetime) else str(r["time"]),
                    r["cell_id"],
                    r["source"],
                    r["data_mode"],
                    r.get("ndvi"),
                    r.get("ndvi_anomaly"),
                    r.get("sar_backscatter"),
                    r.get("soil_moisture"),
                    r.get("lst_anomaly"),
                    r.get("hyperspectral_mn_idx"),
                    r.get("rainfall_mm"),
                    r.get("amt_conductivity"),
                    r.get("ant_velocity_pct"),
                    r.get("insar_deform_mm"),
                )
                for r in records
            ],
        )
        await conn.commit()
    finally:
        await conn.close()


async def upsert_reserve_estimates(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    conn = await get_db_conn()
    try:
        await conn.executemany(
            """
            INSERT INTO reserve_estimates
            (time, cell_id, model_version, ore_prob, ore_uncertainty,
             reserve_class, feature_json)
            VALUES (?,?,?,?,?,?,?)
            """,
            [
                (
                    r["time"].isoformat() if isinstance(r["time"], datetime) else str(r["time"]),
                    r["cell_id"],
                    r["model_version"],
                    r["ore_prob"],
                    r["ore_uncertainty"],
                    r["reserve_class"],
                    json.dumps(r.get("feature_json", {})),
                )
                for r in records
            ],
        )
        await conn.commit()
    finally:
        await conn.close()


async def insert_shortfall_prediction(
    risk_pct: float,
    uncertainty: float,
    top_drivers: List[Dict],
    forecast_json: List[Dict],
    model_version: str,
) -> None:
    conn = await get_db_conn()
    try:
        await conn.execute(
            """
            INSERT INTO shortfall_predictions
            (time, model_version, risk_pct, uncertainty, top_drivers, forecast_json)
            VALUES (?,?,?,?,?,?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                model_version,
                risk_pct,
                uncertainty,
                json.dumps(top_drivers),
                json.dumps(forecast_json),
            ),
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_latest_reserve_estimates() -> List[Dict]:
    conn = await get_db_conn()
    try:
        cursor = await conn.execute(
            """
            SELECT re.cell_id, re.ore_prob, re.ore_uncertainty, re.reserve_class,
                   re.model_version, re.feature_json, re.time,
                   gc.row_idx, gc.col_idx,
                   gc.centroid_lat AS lat, gc.centroid_lon AS lon,
                   gc.elevation_m
            FROM reserve_estimates re
            JOIN grid_cells gc ON gc.id = re.cell_id
            WHERE re.time = (
                SELECT MAX(re2.time) FROM reserve_estimates re2 WHERE re2.cell_id = re.cell_id
            )
            ORDER BY re.cell_id
            """
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def get_latest_shortfall() -> Optional[Dict]:
    conn = await get_db_conn()
    try:
        cursor = await conn.execute(
            """
            SELECT time, model_version, risk_pct, uncertainty,
                   top_drivers, forecast_json
            FROM shortfall_predictions
            ORDER BY time DESC LIMIT 1
            """
        )
        row = await cursor.fetchone()
        if row:
            d = dict(row)
            d["top_drivers"] = json.loads(d["top_drivers"])
            d["forecast_json"] = json.loads(d["forecast_json"])
            return d
        return None
    finally:
        await conn.close()


async def get_latest_iot() -> List[Dict]:
    conn = await get_db_conn()
    try:
        cursor = await conn.execute(
            """
            SELECT machine_id, time, lat, lon,
                   vibration_g, fuel_rate_lph, state, health_score
            FROM iot_telemetry
            WHERE time = (
                SELECT MAX(t2.time) FROM iot_telemetry t2 WHERE t2.machine_id = iot_telemetry.machine_id
            )
            ORDER BY machine_id
            """
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await conn.close()
