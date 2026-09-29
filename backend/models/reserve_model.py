"""
OreSense AI – Reserve Mapping Model (Module 3.1)
================================================
RandomForestRegressor on 7 fused spatial features per grid cell:
  1. hyperspectral_mn_idx   – Mn-oxide spectral index (EnMAP/PRISMA simulated)
  2. ndvi_anomaly           – NDVI deviation from baseline (Sentinel-2)
  3. soil_moisture          – Soil moisture proxy (Sentinel-1 SAR)
  4. insar_deform_mm        – Surface deformation (InSAR, CACHED/simulated)
  5. elevation_m            – Digital elevation model
  6. amt_conductivity       – AMT/CSAMT apparent conductivity (simulated)
  7. ant_velocity_pct       – ANT shear-wave velocity anomaly (simulated)

Output per cell:
  - ore_prob (0–1): predicted ore grade probability
  - ore_uncertainty: std across trees
  - reserve_class: Measured (>0.7) / Indicated (0.5–0.7) / Inferred (0.3–0.5) / Below

Training: synthetic mines generated with varied seeds. Evaluated on
held-out mines. R² reported for surface-only vs surface+subsurface feature sets.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from core.synthetic_mine import build_synthetic_mine, get_mine
from config import settings

WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", str(Path(__file__).resolve().parent / "weights")))
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = WEIGHTS_DIR / "reserve_model.pkl"
MODEL_VERSION_PATH = WEIGHTS_DIR / "reserve_model_version.json"

FEATURE_COLS = [
    "hyperspectral_mn_idx",
    "ndvi_anomaly",
    "soil_moisture",
    "insar_deform_mm",
    "elevation_m",
    "amt_conductivity",
    "ant_velocity_pct",
]
SURFACE_FEATURES = ["hyperspectral_mn_idx", "ndvi_anomaly", "soil_moisture", "elevation_m"]
SUBSURFACE_FEATURES = ["insar_deform_mm", "amt_conductivity", "ant_velocity_pct"]


def _mine_to_df(
    rows: int,
    cols: int,
    seed: int,
    noise_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Build one synthetic mine and return a DataFrame of per-cell features
    and targets suitable for training.
    """
    mine = build_synthetic_mine(rows, cols, seed=seed)
    rng = np.random.default_rng(seed + 9999)

    flat_grade = mine.true_grade.ravel()
    n = len(flat_grade)

    # Simulate each sensor with noise
    hyperspectral = np.clip(
        0.1 + 0.7 * flat_grade + rng.normal(0, 0.04 * noise_scale, n), 0, 1
    )
    ndvi_anomaly = np.clip(
        -0.15 * flat_grade + rng.normal(0, 0.025 * noise_scale, n), -0.5, 0.5
    )
    soil_moisture = np.clip(
        0.25 + 0.1 * flat_grade + rng.normal(0, 0.04 * noise_scale, n), 0, 1
    )
    insar = -2.5 * flat_grade + rng.normal(0, 0.3 * noise_scale, n)
    elevation = mine.amt_conductivity_base.ravel() * 0  # placeholder; from grid
    # Use a simple terrain proxy
    rr, cc = np.mgrid[0:rows, 0:cols]
    elevation = (
        600 + 80 * np.exp(-((rr - rows / 2) ** 2 + (cc - cols / 2) ** 2) / (2 * (rows / 4) ** 2))
    ).ravel()

    amt = np.clip(
        mine.amt_conductivity_base.ravel() + rng.normal(0, 0.8 * noise_scale, n), 0.05, 55
    )
    ant = mine.ant_velocity_pct_base.ravel() + rng.normal(0, 1.5 * noise_scale, n)

    df = pd.DataFrame(
        {
            "hyperspectral_mn_idx": hyperspectral,
            "ndvi_anomaly": ndvi_anomaly,
            "soil_moisture": soil_moisture,
            "insar_deform_mm": insar,
            "elevation_m": elevation,
            "amt_conductivity": amt,
            "ant_velocity_pct": ant,
            "true_grade": flat_grade,
        }
    )
    return df


def generate_training_data(
    rows: int = 20,
    cols: int = 13,
    n_mines: int = 50,
) -> pd.DataFrame:
    """Generate training data from n_mines distinct synthetic mines."""
    frames = [_mine_to_df(rows, cols, seed=i) for i in range(n_mines)]
    return pd.concat(frames, ignore_index=True)


def train(rows: int = 20, cols: int = 13, n_mines: int = 50) -> Dict:
    """
    Train the reserve mapping model.
    Returns metrics dict including R² for surface-only and full feature sets.
    """
    df = generate_training_data(rows, cols, n_mines)

    X_all = df[FEATURE_COLS].values
    X_surf = df[SURFACE_FEATURES].values
    y = df["true_grade"].values

    X_tr_all, X_te_all, X_tr_s, X_te_s, y_tr, y_te = train_test_split(
        X_all, X_surf, y, test_size=0.2, random_state=42
    )

    # Full feature model
    rf_all = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf_all.fit(X_tr_all, y_tr)

    # Surface-only model (for R² comparison)
    rf_surf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf_surf.fit(X_tr_s, y_tr)

    preds_all = rf_all.predict(X_te_all)
    preds_surf = rf_surf.predict(X_te_s)

    metrics = {
        "r2_surface_only": round(float(r2_score(y_te, preds_surf)), 4),
        "r2_surface_plus_subsurface": round(float(r2_score(y_te, preds_all)), 4),
        "mae_all": round(float(mean_absolute_error(y_te, preds_all)), 4),
        "n_training_cells": len(y_tr),
        "n_mines": n_mines,
        "feature_importances": dict(
            zip(FEATURE_COLS, rf_all.feature_importances_.round(4).tolist())
        ),
    }

    # Save full model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(rf_all, f)

    version = _next_version()
    with open(MODEL_VERSION_PATH, "w") as f:
        json.dump({"version": version, "metrics": metrics}, f, indent=2)

    return {"version": version, "metrics": metrics}


def _next_version() -> str:
    import datetime
    return f"reserve_v{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


def load_model() -> Optional[RandomForestRegressor]:
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


def get_version_info() -> Dict:
    if MODEL_VERSION_PATH.exists():
        with open(MODEL_VERSION_PATH) as f:
            return json.load(f)
    return {"version": "untrained", "metrics": {}}


def classify_reserve(prob: float) -> str:
    if prob >= 0.70:
        return "Measured"
    if prob >= 0.50:
        return "Indicated"
    if prob >= 0.30:
        return "Inferred"
    return "Below"


def predict_grid(feature_grid: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Given per-cell feature arrays (rows × cols), return ore_prob and uncertainty.
    feature_grid: dict keyed by FEATURE_COLS
    """
    model = load_model()
    if model is None:
        raise RuntimeError("Reserve model not trained yet")

    rows, cols = list(feature_grid.values())[0].shape
    n = rows * cols
    X = np.column_stack([feature_grid[f].ravel() for f in FEATURE_COLS])

    # Per-tree predictions for uncertainty
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    ore_prob = np.mean(tree_preds, axis=0).clip(0, 1).reshape(rows, cols)
    ore_std = np.std(tree_preds, axis=0).reshape(rows, cols)

    return {"ore_prob": ore_prob, "ore_uncertainty": ore_std}
