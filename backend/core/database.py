"""
OreSense AI – Database helpers (async SQLAlchemy + asyncpg)
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from config import settings

# Use asyncpg driver
_async_url = settings.database_url.replace(
    "postgresql://", "postgresql+asyncpg://"
).replace(
    "postgresql+psycopg2://", "postgresql+asyncpg://"
)

engine = create_async_engine(_async_url, pool_pre_ping=True, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_raw_conn() -> asyncpg.Connection:
    """Raw asyncpg connection for bulk inserts."""
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    return await asyncpg.connect(dsn)


async def bulk_insert_observations(records: List[Dict[str, Any]]) -> None:
    """
    Bulk-insert remote-sensing observations into the rs_observations table.
    Each record must have: time, cell_id, source, data_mode, and optional fields.
    """
    if not records:
        return
    conn = await get_raw_conn()
    try:
        await conn.executemany(
            """
            INSERT INTO rs_observations
            (time, cell_id, source, data_mode,
             ndvi, ndvi_anomaly, sar_backscatter, soil_moisture,
             lst_anomaly, hyperspectral_mn_idx, rainfall_mm,
             amt_conductivity, ant_velocity_pct, insar_deform_mm)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
            ON CONFLICT DO NOTHING
            """,
            [
                (
                    r["time"],
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
    finally:
        await conn.close()


async def upsert_reserve_estimates(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    conn = await get_raw_conn()
    try:
        await conn.executemany(
            """
            INSERT INTO reserve_estimates
            (time, cell_id, model_version, ore_prob, ore_uncertainty,
             reserve_class, feature_json)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            """,
            [
                (
                    r["time"],
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
    finally:
        await conn.close()


async def insert_shortfall_prediction(
    risk_pct: float,
    uncertainty: float,
    top_drivers: List[Dict],
    forecast_json: List[Dict],
    model_version: str,
) -> None:
    conn = await get_raw_conn()
    try:
        await conn.execute(
            """
            INSERT INTO shortfall_predictions
            (time, model_version, risk_pct, uncertainty, top_drivers, forecast_json)
            VALUES ($1,$2,$3,$4,$5,$6)
            """,
            datetime.now(timezone.utc),
            model_version,
            risk_pct,
            uncertainty,
            json.dumps(top_drivers),
            json.dumps(forecast_json),
        )
    finally:
        await conn.close()


async def get_latest_reserve_estimates() -> List[Dict]:
    conn = await get_raw_conn()
    try:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (cell_id)
                re.cell_id, re.ore_prob, re.ore_uncertainty, re.reserve_class,
                re.model_version, re.feature_json, re.time,
                gc.row_idx, gc.col_idx,
                ST_Y(gc.centroid) AS lat, ST_X(gc.centroid) AS lon,
                gc.elevation_m
            FROM reserve_estimates re
            JOIN grid_cells gc ON gc.id = re.cell_id
            ORDER BY cell_id, re.time DESC
            """
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def get_latest_shortfall() -> Optional[Dict]:
    conn = await get_raw_conn()
    try:
        row = await conn.fetchrow(
            """
            SELECT time, model_version, risk_pct, uncertainty,
                   top_drivers, forecast_json
            FROM shortfall_predictions
            ORDER BY time DESC LIMIT 1
            """
        )
        if row:
            d = dict(row)
            d["top_drivers"] = json.loads(d["top_drivers"])
            d["forecast_json"] = json.loads(d["forecast_json"])
            return d
        return None
    finally:
        await conn.close()


async def get_latest_iot() -> List[Dict]:
    conn = await get_raw_conn()
    try:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (machine_id)
                machine_id, time, lat, lon,
                vibration_g, fuel_rate_lph, state, health_score
            FROM iot_telemetry
            ORDER BY machine_id, time DESC
            """
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()
