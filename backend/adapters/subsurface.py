"""
OreSense AI – Subsurface Adapters (AMT/CSAMT + ANT)
=====================================================
Both adapters are ALWAYS SIMULATED – geophysical surveys require physical
field instruments and cannot be fetched from a web API.

AMT (Audio-Magnetotellurics):
  Per-cell apparent conductivity (mS/m) derived from the synthetic ore body.
  Ore (magnetite, pyrolusite) is more conductive than host granite/quartzite.

ANT (Ambient Noise Tomography):
  Per-cell shear-wave velocity anomaly (%) – negative over ore zones where the
  host rock is fractured, altered, or saturated.

Noise model: multiplicative Gaussian (5%) + additive Gaussian (variable).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from core.adapter import DataAdapter, DataMode, Observation
from core.synthetic_mine import get_mine
from config import settings


class AMTAdapter(DataAdapter):
    source_name = "amt_csamt"
    default_mode = DataMode.SIMULATED

    async def fetch_live(self, **kwargs) -> Observation:
        raise NotImplementedError("AMT requires physical field instruments")

    async def fetch_simulated(self, **kwargs) -> Observation:
        mine = get_mine(settings.grid_rows, settings.grid_cols)
        rng = np.random.default_rng(int(datetime.utcnow().timestamp() / 3600) % 100000)

        # Base conductivity from ore model
        base = mine.amt_conductivity_base  # 0.1–50 mS/m
        # Multiplicative noise (instrument + lateral heterogeneity)
        mult_noise = rng.normal(1.0, 0.05, base.shape)
        add_noise = rng.normal(0.0, 0.8, base.shape)
        conductivity = np.clip(base * mult_noise + add_noise, 0.05, 55.0).astype(np.float32)

        return Observation(
            source=self.source_name,
            mode=DataMode.SIMULATED,
            acquired_at=datetime.now(timezone.utc),
            data={"amt_conductivity": conductivity.tolist()},
            metadata={
                "units": "mS/m",
                "note": (
                    "SIMULATED – AMT/CSAMT conductivity derived from synthetic ore model. "
                    "Physical basis: sulphide/oxide ores are 100–10000x more conductive "
                    "than silicate host rock."
                ),
            },
        )


class ANTAdapter(DataAdapter):
    source_name = "ant_tomography"
    default_mode = DataMode.SIMULATED

    async def fetch_live(self, **kwargs) -> Observation:
        raise NotImplementedError("ANT requires seismic sensor network processing")

    async def fetch_simulated(self, **kwargs) -> Observation:
        mine = get_mine(settings.grid_rows, settings.grid_cols)
        rng = np.random.default_rng(int(datetime.utcnow().timestamp() / 3600) % 100000)

        base = mine.ant_velocity_pct_base  # 0 to -25 %
        noise = rng.normal(0.0, 1.5, base.shape)
        velocity_pct = (base + noise).astype(np.float32)

        return Observation(
            source=self.source_name,
            mode=DataMode.SIMULATED,
            acquired_at=datetime.now(timezone.utc),
            data={"ant_velocity_pct": velocity_pct.tolist()},
            metadata={
                "units": "%",
                "note": (
                    "SIMULATED – ANT shear-wave velocity anomaly from synthetic model. "
                    "Physical basis: fracturing and alteration reduce Vs by 5–25%."
                ),
            },
        )


class HyperspectralAdapter(DataAdapter):
    """
    EnMAP / PRISMA Mn-oxide absorption index.
    Always SIMULATED – public sample scenes are not available for this AOI.
    Physical basis: Mn-oxides (pyrolusite, birnessite) have diagnostic
    absorption features at 480 nm and 780 nm.
    Index = (R480 - R780) / (R480 + R780), normalised to 0–1.
    """
    source_name = "hyperspectral_enmap"
    default_mode = DataMode.SIMULATED

    async def fetch_live(self, **kwargs) -> Observation:
        raise NotImplementedError("No public EnMAP scene available for Balaghat AOI")

    async def fetch_simulated(self, **kwargs) -> Observation:
        mine = get_mine(settings.grid_rows, settings.grid_cols)
        rng = np.random.default_rng(int(datetime.utcnow().timestamp() / 3600) % 100000)

        # Mn index scales with ore grade; noise simulates atmospheric + sensor noise
        mn_idx = (0.1 + 0.7 * mine.true_grade + rng.normal(0, 0.04, mine.true_grade.shape))
        mn_idx = np.clip(mn_idx, 0.0, 1.0).astype(np.float32)

        acquired_at = datetime.now(timezone.utc)

        return Observation(
            source=self.source_name,
            mode=DataMode.SIMULATED,
            acquired_at=acquired_at,
            data={"hyperspectral_mn_idx": mn_idx.tolist()},
            metadata={
                "note": (
                    "SIMULATED – EnMAP/PRISMA Mn-oxide spectral index. "
                    "No real hyperspectral scene exists for this AOI. "
                    "Absorption features at 480 nm and 780 nm are diagnostic "
                    "of pyrolusite and birnessite."
                ),
                "bands_used": "480nm, 780nm (simulated)",
            },
        )
