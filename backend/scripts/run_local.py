"""
OreSense AI – Local Runner (No Docker)
========================================
Initialises a SQLite database, seeds data, trains models, and starts the
FastAPI server. Replaces the Docker Compose workflow for local development.

Usage:
    python scripts/run_local.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Set up the Python path to the backend directory
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

# Force simulated mode and set env vars before importing config
os.environ.setdefault("MODE", "SIMULATED")
os.environ["DATABASE_URL"] = "sqlite:///data/oresense.db"
os.environ.setdefault("MQTT_BROKER", "localhost")
os.environ.setdefault("MQTT_PORT", "1883")

from config import settings


def banner(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


async def seed_grid():
    """Seed the spatial grid into SQLite."""
    from core.database_local import get_db_conn
    from core.grid import build_grid_gdf

    print("[Seed] Building spatial grid...")
    gdf = build_grid_gdf(
        settings.aoi_minlat, settings.aoi_maxlat,
        settings.aoi_minlon, settings.aoi_maxlon,
        settings.grid_rows, settings.grid_cols,
    )

    conn = await get_db_conn()
    try:
        for _, row in gdf.iterrows():
            centroid = row["centroid"]
            await conn.execute(
                """
                INSERT OR IGNORE INTO grid_cells (row_idx, col_idx, centroid_lat, centroid_lon, elevation_m)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(row["row_idx"]),
                    int(row["col_idx"]),
                    float(centroid.y),
                    float(centroid.x),
                    float(row["elevation_m"]),
                ),
            )
        await conn.commit()
        print(f"[Seed] Grid: {len(gdf)} cells inserted")
    finally:
        await conn.close()


async def seed_production():
    """Seed production records into SQLite."""
    from core.database_local import get_db_conn
    from simulators.production_sim import generate_production_records

    print("[Seed] Generating 2 years of production records...")
    df = generate_production_records(days=730)

    conn = await get_db_conn()
    try:
        for _, row in df.iterrows():
            await conn.execute(
                """
                INSERT OR IGNORE INTO production_records
                (date, site, planned_tonnes, actual_tonnes, shortfall_pct,
                 rainfall_mm, equipment_downtime_h, crew_available, blasting_delays_h)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
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
                ),
            )
        await conn.commit()
        print(f"[Seed] Production: {len(df)} records inserted")
    finally:
        await conn.close()


async def seed_initial_observations():
    """Run the ingestion pipeline once to seed RS observation data."""
    print("[Seed] Running initial ingestion pass...")
    try:
        from pipeline.ingestion import ingest_all
        result = await ingest_all()
        print(f"[Seed] Initial ingestion: {result.get('n_cells', 0)} cells")
    except Exception as e:
        print(f"[Seed] Initial ingestion failed (non-fatal): {e}")


def train_all():
    """Train all ML models (synchronous)."""
    import warnings
    warnings.filterwarnings("ignore")

    # 1. Reserve model
    print("\n[1/4] Training Reserve Mapping Model...")
    try:
        from models.reserve_model import train as train_reserve
        result = train_reserve(
            rows=settings.grid_rows,
            cols=settings.grid_cols,
            n_mines=30,
        )
        print(f"  R² (surface only):          {result['metrics']['r2_surface_only']:.4f}")
        print(f"  R² (surface + subsurface):  {result['metrics']['r2_surface_plus_subsurface']:.4f}")
        print(f"  Version: {result['version']}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 2. Shortfall model
    print("\n[2/4] Training Shortfall Prediction Model...")
    try:
        from models.shortfall_model import train as train_shortfall
        result = train_shortfall()
        print(f"  R²: {result['metrics']['r2']:.4f}  MAE: {result['metrics']['mae']:.2f}%")
        print(f"  Version: {result['version']}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 3. Causal model
    print("\n[3/4] Running Causal Analysis...")
    try:
        from models.causal_model import run_causal_analysis
        from simulators.production_sim import generate_production_records
        df = generate_production_records(days=730)
        results = run_causal_analysis(df)
        mediation = results.get("mediation_summary", {})
        prop_med = mediation.get("proportion_mediated", 0)
        print(f"  Proportion of rainfall effect mediated by downtime: {prop_med:.1%}")

        # Store causal results in SQLite
        import sqlite3
        from core.database_local import DB_PATH
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "INSERT INTO causal_estimates (model_version, results_json) VALUES (?, ?)",
            ("causal_v1", json.dumps(results)),
        )
        conn.commit()
        conn.close()
        print("  Causal results stored in DB")
    except Exception as e:
        print(f"  ERROR (non-fatal): {e}")

    # 4. RL Digital Twin
    rl_timesteps = int(os.getenv("RL_TIMESTEPS", "50000"))
    print(f"\n[4/4] Training RL Digital Twin ({rl_timesteps} timesteps)...")
    try:
        from models.rl_twin import train_rl
        result = train_rl(timesteps=rl_timesteps)
        print(f"  {result}")
    except Exception as e:
        print(f"  ERROR (non-fatal, RL optional): {e}")


async def seed_all():
    """Seed database with grid, production, and initial observation data."""
    await seed_grid()
    await seed_production()
    await seed_initial_observations()


def main():
    banner("OreSense AI – Local Setup (No Docker)")

    # Step 1: Init SQLite DB
    print("[1/3] Initialising SQLite database...")
    from core.database_local import init_db_sync, DB_PATH
    init_db_sync()
    print(f"  Database: {DB_PATH}")

    # Step 2: Seed data
    print("\n[2/3] Seeding data...")
    asyncio.run(seed_all())

    # Step 3: Train models
    banner("Training All Models")
    train_all()

    # Step 4: Register models in DB
    print("\n[Registering models in DB...]")
    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        from models.reserve_model import get_version_info as rv
        from models.shortfall_model import get_version_info as sv
        rinfo = rv()
        sinfo = sv()
        conn.execute(
            "INSERT OR IGNORE INTO model_registry (model_name, version, metrics_json, is_active) VALUES (?,?,?,1)",
            ("reserve_mapping", rinfo.get("version", "unknown"), json.dumps(rinfo.get("metrics", {}))),
        )
        conn.execute(
            "INSERT OR IGNORE INTO model_registry (model_name, version, metrics_json, is_active) VALUES (?,?,?,1)",
            ("shortfall_prediction", sinfo.get("version", "unknown"), json.dumps(sinfo.get("metrics", {}))),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  Model registration failed (non-fatal): {e}")

    # Step 5: Seed FL rounds (demo data)
    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        for i in range(1, 6):
            conn.execute(
                "INSERT OR IGNORE INTO fl_rounds (round_num, global_accuracy, client_metrics) VALUES (?,?,?)",
                (i, 0.85 + 0.02 * i, json.dumps([
                    {"site": "Balaghat", "accuracy": 0.83 + 0.02 * i, "loss": 0.3 - 0.03 * i},
                    {"site": "Nagpur", "accuracy": 0.81 + 0.02 * i, "loss": 0.32 - 0.03 * i},
                    {"site": "Gumgaon", "accuracy": 0.80 + 0.02 * i, "loss": 0.34 - 0.03 * i},
                ])),
            )
        conn.commit()
        conn.close()
        print("  FL demo rounds seeded")
    except Exception as e:
        print(f"  FL seeding failed (non-fatal): {e}")

    banner("Setup Complete!")
    print("Starting FastAPI server...")
    print("  Dashboard: http://localhost:3000")
    print("  API docs:  http://localhost:8000/docs")
    print()

    # Start the server
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(BACKEND_DIR)],
    )


if __name__ == "__main__":
    main()
