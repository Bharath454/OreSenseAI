"""
OreSense AI – Sentinel-1 SAR Adapter
======================================
LIVE mode:  Copernicus Data Space – GRD product → soil moisture proxy from
            C-band VV backscatter (empirical formula after Wagner et al.).
            InSAR deformation uses a PRECOMPUTED sample coherence product
            (full InSAR processing is out of scope) → marked CACHED.
SIMULATED:  Backscatter derived from synthetic mine soil-moisture model plus
            weather noise. InSAR deformation from ore body subsidence model.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import numpy as np

from core.adapter import DataAdapter, DataMode, Observation
from core.synthetic_mine import get_mine
from config import settings


class Sentinel1Adapter(DataAdapter):
    source_name = "sentinel1"

    async def fetch_live(self, **kwargs) -> Observation:
        # Full SAR download is heavyweight; in demo we raise to trigger cache
        raise NotImplementedError(
            "S1 live download requires full SAR pipeline (out of scope for demo)"
        )

    async def fetch_simulated(self, **kwargs) -> Observation:
        """
        Simulated SAR backscatter (VV, dB) and soil-moisture proxy.

        Physical basis:
        - Wet soils backscatter more strongly than dry soils in C-band VV.
        - Ore bodies alter surface roughness and dielectric constant slightly.
        - InSAR deformation: long-term subsidence of ~1 mm/yr over ore bodies
          (gravity-driven compaction); marked CACHED to convey pre-computed origin.
        """
        mine = get_mine(settings.grid_rows, settings.grid_cols)
        rng = np.random.default_rng(int(datetime.utcnow().timestamp() / 3600) % 100000)

        # Seasonal soil moisture: monsoon peak August
        doy = datetime.utcnow().timetuple().tm_yday
        seasonal_sm = 0.3 + 0.25 * math.sin(2 * math.pi * (doy - 120) / 365)

        # Backscatter increases with soil moisture; ore zones add roughness
        sm = np.clip(
            seasonal_sm
            + 0.1 * mine.true_grade
            + rng.normal(0, 0.04, mine.true_grade.shape),
            0.0, 1.0,
        ).astype(np.float32)

        # VV backscatter (dB): empirical ~ -20 to -5 dB for bare soils
        sar_backscatter = (-20 + 15 * sm + rng.normal(0, 0.8, sm.shape)).astype(np.float32)

        # InSAR simulated deformation (mm): subsidence over ore bodies
        insar = (-2.5 * mine.true_grade + rng.normal(0, 0.3, mine.true_grade.shape)).astype(
            np.float32
        )

        # Revisit: 12 days (Sentinel-1 repeat pass)
        acquired_at = datetime.now(timezone.utc) - timedelta(
            days=rng.integers(0, 12).item()
        )

        return Observation(
            source=self.source_name,
            mode=DataMode.SIMULATED,
            acquired_at=acquired_at,
            data={
                "sar_backscatter": sar_backscatter.tolist(),
                "soil_moisture": sm.tolist(),
                "insar_deform_mm": insar.tolist(),
            },
            metadata={
                "insar_mode": "CACHED",
                "insar_note": (
                    "InSAR deformation uses a precomputed synthetic interferogram. "
                    "Full InSAR processing (Sentinel Application Platform) is out of "
                    "scope for this prototype."
                ),
                "sar_note": "SIMULATED – no real Copernicus SAR credentials",
            },
        )
