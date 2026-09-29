"""
OreSense AI – Model Training Script
Trains reserve mapping, shortfall prediction, causal model, and RL twin.
Run once at startup after seeding.
"""
import sys
import os
import json
import asyncio

sys.path.insert(0, "/app")
os.environ.setdefault("MODE", "SIMULATED")

import asyncpg
from config import settings


async def train_all():
    print("=" * 60)
    print("OreSense AI – Training All Models")
    print("=" * 60)

    # 1. Reserve model
    print("\n[1/4] Training Reserve Mapping Model...")
    try:
        from models.reserve_model import train as train_reserve
        result = train_reserve(
            rows=settings.grid_rows,
            cols=settings.grid_cols,
            n_mines=30,  # reduced for fast startup; increase for better accuracy
        )
        print(f"  R² (surface only):          {result['metrics']['r2_surface_only']:.4f}")
        print(f"  R² (surface + subsurface):  {result['metrics']['r2_surface_plus_subsurface']:.4f}")
        print(f"  Version: {result['version']}")

        # Register in DB
        conn = await asyncpg.connect(settings.database_url)
        await conn.execute(
            """
            INSERT INTO model_registry (model_name, version, metrics_json, is_active)
            VALUES ($1, $2, $3, TRUE)
            ON CONFLICT DO NOTHING
            """,
            "reserve_mapping",
            result["version"],
            json.dumps(result["metrics"]),
        )
        await conn.close()
    except Exception as e:
        print(f"  ERROR: {e}")

    # 2. Shortfall model
    print("\n[2/4] Training Shortfall Prediction Model...")
    try:
        from models.shortfall_model import train as train_shortfall
        result = train_shortfall()
        print(f"  R²: {result['metrics']['r2']:.4f}  MAE: {result['metrics']['mae']:.2f}%")
        print(f"  Version: {result['version']}")

        conn = await asyncpg.connect(settings.database_url)
        await conn.execute(
            """
            INSERT INTO model_registry (model_name, version, metrics_json, is_active)
            VALUES ($1, $2, $3, TRUE)
            ON CONFLICT DO NOTHING
            """,
            "shortfall_prediction",
            result["version"],
            json.dumps(result["metrics"]),
        )
        await conn.close()
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
        print(f"  Interpretation: {mediation.get('interpretation', '')}")

        conn = await asyncpg.connect(settings.database_url)
        await conn.execute(
            """
            INSERT INTO causal_estimates (model_version, results_json)
            VALUES ($1, $2)
            """,
            "causal_v1",
            json.dumps(results),
        )
        await conn.close()
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

    print("\n[Done] All models trained successfully.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(train_all())
