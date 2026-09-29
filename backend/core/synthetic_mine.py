"""
OreSense AI – Synthetic Mine Generator
=======================================
Creates a hidden ground-truth ore body used as the true signal for ALL
simulated data. No real MOIL data is used – this is a synthetic prototype.

Design
------
- 3 elliptical ore bodies with Gaussian falloff placed randomly in the grid.
- True ore grade (0–1) is the sum of individual contributions, clipped to [0,1].
- All sensor simulators add Gaussian noise ON TOP of signals derived from this.
- This object is instantiated once per process and cached.

IMPORTANT: This is a purely synthetic model. It does NOT represent real ore
           locations in Balaghat or any other MOIL property.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "synthetic_mine.json"


@dataclass
class OreBody:
    row_center: float
    col_center: float
    row_sigma: float
    col_sigma: float
    peak_grade: float   # 0–1
    angle_deg: float    # rotation of ellipse


@dataclass
class SyntheticMine:
    rows: int
    cols: int
    ore_bodies: List[OreBody]
    true_grade: np.ndarray   # shape (rows, cols), 0–1

    # Derived physical properties (noiseless)
    # Higher grade → higher conductivity (ore is more conductive than host rock)
    amt_conductivity_base: np.ndarray   # mS/m, shape (rows, cols)
    # Higher grade → lower shear-wave velocity (pore space, alteration)
    ant_velocity_pct_base: np.ndarray   # % anomaly, shape (rows, cols)
    # Surface expression: mild NDVI suppression over ore (vegetation stress)
    ndvi_suppression: np.ndarray        # 0–0.15, shape (rows, cols)


def _gaussian_ellipse(
    rows: int, cols: int, body: OreBody
) -> np.ndarray:
    """Return a 2-D Gaussian on a (rows × cols) grid, rotated by angle_deg."""
    rr, cc = np.mgrid[0:rows, 0:cols].astype(float)
    dr = rr - body.row_center
    dc = cc - body.col_center
    theta = np.radians(body.angle_deg)
    dr_rot = dr * np.cos(theta) + dc * np.sin(theta)
    dc_rot = -dr * np.sin(theta) + dc * np.cos(theta)
    exponent = (
        (dr_rot ** 2) / (2 * body.row_sigma ** 2)
        + (dc_rot ** 2) / (2 * body.col_sigma ** 2)
    )
    return body.peak_grade * np.exp(-exponent)


def build_synthetic_mine(rows: int, cols: int, seed: int = 42) -> SyntheticMine:
    """
    Build a synthetic mine on a (rows × cols) grid.
    The ore bodies are placed deterministically from 'seed'.
    """
    rng = np.random.default_rng(seed)

    # Place 3 ore bodies
    n_bodies = 3
    bodies: List[OreBody] = []
    for _ in range(n_bodies):
        bodies.append(
            OreBody(
                row_center=rng.uniform(2, rows - 2),
                col_center=rng.uniform(2, cols - 2),
                row_sigma=rng.uniform(1.5, 4.0),
                col_sigma=rng.uniform(1.0, 3.0),
                peak_grade=rng.uniform(0.6, 0.95),
                angle_deg=rng.uniform(0, 180),
            )
        )

    # Sum Gaussian falloffs → true ore grade
    true_grade = np.zeros((rows, cols), dtype=np.float32)
    for b in bodies:
        true_grade += _gaussian_ellipse(rows, cols, b).astype(np.float32)
    true_grade = np.clip(true_grade, 0, 1)

    # Derived physical properties
    # AMT: host rock ~0.1 mS/m; ore (magnetite/pyrolusite) ~10–50 mS/m
    amt_conductivity_base = 0.1 + 49.9 * true_grade  # range 0.1–50 mS/m

    # ANT: shear-wave velocity anomaly is negative over ore
    # (fractured, altered host rock), roughly -5% to -25%
    ant_velocity_pct_base = -25.0 * true_grade  # range 0 to -25 %

    # NDVI suppression: Mn toxicity in soils → slight vegetation stress
    ndvi_suppression = 0.15 * true_grade

    return SyntheticMine(
        rows=rows,
        cols=cols,
        ore_bodies=bodies,
        true_grade=true_grade,
        amt_conductivity_base=amt_conductivity_base,
        ant_velocity_pct_base=ant_velocity_pct_base,
        ndvi_suppression=ndvi_suppression,
    )


# ── Singleton ─────────────────────────────────────────────────────────────────
_mine: SyntheticMine | None = None


def get_mine(rows: int = 20, cols: int = 13, seed: int = 42) -> SyntheticMine:
    global _mine
    if _mine is None:
        _mine = build_synthetic_mine(rows, cols, seed)
    return _mine
