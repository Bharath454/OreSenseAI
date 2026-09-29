"""
OreSense AI – Database Seeding Script
Populates grid_cells, production_records, and initial rs_observations.
Run once at startup before training.
"""
import asyncio
import sys
import os

sys.path.insert(0, "/app")
os.environ.setdefault("MODE", "SIMULATED")

import asyncpg
import numpy as np
from datetime import timezone

from config import settings
from core.grid import build_grid_gdf, cell_id
from simulators.production_sim import generate_production_records

DATABASE_URL = settings.database_url


async def seed_grid(conn: asyncpg.Connection) -> None:
    print("[Seed] Building spatial grid...")
    gdf = build_grid_gdf(
        settings.aoi_minlat, settings.aoi_maxlat,
        settings.aoi_minlon, settings.aoi_maxlon,
        settings.grid_rows, settings.grid_cols,
    )
    records = []
    for _, row in gdf.iterrows():
        geom_wkt = row["geom"].wkt
        cent_wkt = row["centroid"].wkt
        records.append((
            int(row["row_idx"]),
            int(row["col_idx"]),
            geom_wkt,
            cent_wkt,
            float(row["elevation_m"]),
        ))

    await conn.executemany(
        """
        INSERT INTO grid_cells (row_idx, col_idx, geom, centroid, elevation_m)
        VALUES ($1, $2, ST_GeomFromText($3, 4326), ST_GeomFromText($4, 4326), $5)
        ON CONFLICT (row_idx, col_idx) DO NOTHING
        """,
        records,
    )
    print(f"[Seed] Grid: {len(records)} cells inserted")


async def seed_production(conn: asyncpg.Connection) -> None:
    print("[Seed] Generating 2 years of production records...")
    df = generate_production_records(days=730)
    records = [
        (
            str(row["date"]),
            row["site"],
            row["planned_tonnes"],
            row["actual_tonnes"],
            row["shortfall_pct"],
            row["rainfall_mm"],
            row["equipment_downtime_h"],
            row["crew_available"],
            row["blasting_delays_h"],
        )
        for _, row in df.iterrows()
    ]
    await conn.executemany(
        """
        INSERT INTO production_records
        (date, site, planned_tonnes, actual_tonnes, shortfall_pct,
         rainfall_mm, equipment_downtime_h, crew_available, blasting_delays_h)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        ON CONFLICT (date) DO NOTHING
        """,
        records,
    )
    print(f"[Seed] Production: {len(records)} records inserted")


async def seed_initial_observations(conn: asyncpg.Connection) -> None:
    """Run the ingestion pipeline once synchronously to seed initial RS data."""
    print("[Seed] Running initial ingestion pass...")
    try:
        from pipeline.ingestion import ingest_all
        result = await ingest_all()
        print(f"[Seed] Initial ingestion: {result.get('n_cells', 0)} cells")
    except Exception as e:
        print(f"[Seed] Initial ingestion failed (non-fatal): {e}")


async def main():
    print(f"[Seed] Connecting to {DATABASE_URL[:40]}...")
    for attempt in range(10):
        try:
            conn = await asyncpg.connect(DATABASE_URL)
            break
        except Exception as e:
            print(f"[Seed] DB not ready (attempt {attempt}): {e}")
            await asyncio.sleep(3)
    else:
        print("[Seed] FATAL: could not connect to DB")
        sys.exit(1)

    try:
        await seed_grid(conn)
        await seed_production(conn)
    finally:
        await conn.close()

    # Run ingestion separately (needs its own connection pool)
    await seed_initial_observations(None)
    print("[Seed] Done.")


if __name__ == "__main__":
    asyncio.run(main())
