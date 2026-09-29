"""
OreSense AI – Production & Operations Data Generator
=====================================================
Generates 2 years of synthetic daily production records for MOIL's
Balaghat mine, with:
  - Seasonal production patterns (monsoon slowdown Jun–Sep)
  - Correlated shortfall events driven by EXACTLY the causal factors
    the causal model must identify:
      * heavy rainfall → equipment downtime → shortfall
      * blasting delays → shortfall
      * crew shortage → shortfall
  - Realistic noise

This is the ground-truth training dataset for the shortfall predictor
and causal analysis model.

IMPORTANT: This is entirely synthetic. No real MOIL production records
           are used or replicated.
"""
from __future__ import annotations

import json
import math
import random
from datetime import date, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd


# Monthly production capacity factor (1.0 = full capacity ~1000 T/day)
MONTHLY_CAPACITY = {
    1: 0.95, 2: 0.95, 3: 0.92, 4: 0.90, 5: 0.88,
    6: 0.65, 7: 0.55, 8: 0.58, 9: 0.70, 10: 0.85,
    11: 0.90, 12: 0.93,
}
BASE_PLANNED_TONNES = 1000.0


def generate_production_records(
    start_date: date | None = None,
    days: int = 730,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate `days` of synthetic daily production records.

    Causal structure encoded (ground truth):
      rainfall → equipment_downtime  (coefficient ~0.8)
      equipment_downtime → shortfall (coefficient ~0.6)
      blasting_delays → shortfall    (coefficient ~0.4)
      crew_available → shortfall     (coefficient ~-0.5, negative = more crew = less shortfall)
      rainfall ↛ shortfall directly  (only via downtime)

    Returns a DataFrame with columns:
        date, site, planned_tonnes, actual_tonnes, shortfall_pct,
        rainfall_mm, equipment_downtime_h, crew_available, blasting_delays_h
    """
    if start_date is None:
        start_date = date.today() - timedelta(days=days)

    rng = np.random.default_rng(seed)
    records: List[Dict] = []

    for i in range(days):
        d = start_date + timedelta(days=i)
        month = d.month
        cap = MONTHLY_CAPACITY[month]
        planned = BASE_PLANNED_TONNES * cap

        # ── Exogenous variables ───────────────────────────────────────────────
        # Rainfall: monsoon peak Jun–Sep
        base_rain = {6: 12, 7: 20, 8: 18, 9: 10}.get(month, 1.5)
        rain_heavy = float(rng.exponential(base_rain * 2)) if rng.random() < 0.2 else 0.0
        rainfall_mm = max(0.0, float(rng.gamma(1.5, base_rain / 1.5)) + rain_heavy)

        # Crew: occasional strikes/absences
        base_crew = int(rng.normal(120, 8))
        if rng.random() < 0.05:   # crew shortage event
            base_crew = int(rng.uniform(60, 85))
        crew = max(40, min(160, base_crew))

        # ── Causal mechanism: rainfall → equipment downtime ───────────────────
        # Empirical: 10 mm of rain ~ 2 h of downtime (slippery haul roads)
        downtime_from_rain = 0.8 * (rainfall_mm / 10) * 2.0
        downtime_noise = float(rng.exponential(0.5))
        equipment_downtime_h = max(0.0, downtime_from_rain + downtime_noise)
        if rng.random() < 0.03:   # random breakdown
            equipment_downtime_h += float(rng.uniform(4, 10))

        # ── Blasting delays ───────────────────────────────────────────────────
        blasting_delays_h = 0.0
        if rng.random() < 0.15:   # 15% chance of blasting issue any day
            blasting_delays_h = float(rng.uniform(1, 6))

        # ── Actual production ─────────────────────────────────────────────────
        # Shortfall mechanism:
        #   downtime reduces production by ~50 T/h
        #   blasting delay reduces production by ~80 T/h
        #   crew shortage reduces by (120-crew) * 3 T/person
        #   rainfall has NO direct effect – only through downtime
        shortfall_from_downtime = 50 * equipment_downtime_h
        shortfall_from_blasting = 80 * blasting_delays_h
        shortfall_from_crew = max(0.0, (120 - crew) * 3.5)
        total_shortfall = (
            shortfall_from_downtime + shortfall_from_blasting + shortfall_from_crew
        )
        noise = float(rng.normal(0, 30))
        actual = max(0.0, planned - total_shortfall + noise)
        shortfall_pct = max(0.0, (planned - actual) / planned * 100)

        records.append(
            {
                "date": d,
                "site": "Balaghat",
                "planned_tonnes": round(planned, 1),
                "actual_tonnes": round(actual, 1),
                "shortfall_pct": round(shortfall_pct, 2),
                "rainfall_mm": round(rainfall_mm, 2),
                "equipment_downtime_h": round(equipment_downtime_h, 2),
                "crew_available": crew,
                "blasting_delays_h": round(blasting_delays_h, 2),
            }
        )

    return pd.DataFrame(records)


def get_production_df(days: int = 730) -> pd.DataFrame:
    return generate_production_records(days=days)
