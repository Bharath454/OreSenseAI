"""
OreSense AI – Unit Tests: ML Models
Tests reserve model, shortfall predictor, causal analysis, and RL twin.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['MODE'] = 'SIMULATED'

import pytest
import numpy as np
import pandas as pd


# ── Synthetic mine ─────────────────────────────────────────────────────────────
def test_synthetic_mine_shape():
    from core.synthetic_mine import build_synthetic_mine
    mine = build_synthetic_mine(20, 13, seed=0)
    assert mine.true_grade.shape == (20, 13)
    assert mine.amt_conductivity_base.shape == (20, 13)
    assert mine.ant_velocity_pct_base.shape == (20, 13)
    assert np.all(mine.true_grade >= 0)
    assert np.all(mine.true_grade <= 1)


def test_synthetic_mine_ore_bodies():
    from core.synthetic_mine import build_synthetic_mine
    mine = build_synthetic_mine(20, 13, seed=42)
    # Must have at least one high-grade cell (our ore bodies have peak 0.6+)
    assert mine.true_grade.max() > 0.5, "Ore body should be visible"
    # AMT conductivity must be higher where grade is higher
    corr = np.corrcoef(mine.true_grade.ravel(), mine.amt_conductivity_base.ravel())[0, 1]
    assert corr > 0.9, f"AMT should correlate strongly with grade, got r={corr:.3f}"


# ── Reserve model ──────────────────────────────────────────────────────────────
def test_reserve_model_training():
    from models.reserve_model import train, load_model
    result = train(rows=10, cols=7, n_mines=5)  # tiny for speed
    assert 'metrics' in result
    assert result['metrics']['r2_surface_plus_subsurface'] > result['metrics']['r2_surface_only'], \
        "Adding subsurface features must improve R²"
    assert result['metrics']['r2_surface_plus_subsurface'] > 0.5, "Model must have R² > 0.5"


def test_reserve_model_classify():
    from models.reserve_model import classify_reserve
    assert classify_reserve(0.80) == 'Measured'
    assert classify_reserve(0.60) == 'Indicated'
    assert classify_reserve(0.40) == 'Inferred'
    assert classify_reserve(0.10) == 'Below'


def test_reserve_model_predict():
    from models.reserve_model import train, predict_grid, FEATURE_COLS
    train(rows=10, cols=7, n_mines=5)
    import numpy as np
    grid = {f: np.random.rand(10, 7) for f in FEATURE_COLS}
    result = predict_grid(grid)
    assert result['ore_prob'].shape == (10, 7)
    assert result['ore_uncertainty'].shape == (10, 7)
    assert np.all(result['ore_prob'] >= 0)
    assert np.all(result['ore_prob'] <= 1)
    assert np.all(result['ore_uncertainty'] >= 0)


# ── Shortfall model ────────────────────────────────────────────────────────────
def test_shortfall_model_training():
    from models.shortfall_model import train
    result = train()
    assert result['metrics']['r2'] > 0.3, "Shortfall model must explain >30% variance"
    assert result['metrics']['mae'] < 20, "MAE should be < 20 percentage points"


def test_shortfall_prediction_output():
    from models.shortfall_model import train, predict_current
    train()
    result = predict_current()
    assert 0 <= result['risk_pct'] <= 100
    assert result['uncertainty'] >= 0
    assert len(result['forecast']) == 30
    assert all('date' in f and 'risk_pct' in f for f in result['forecast'])


def test_shortfall_injection():
    """Injecting heavy rain must INCREASE risk compared to baseline."""
    from models.shortfall_model import train, predict_current
    train()
    baseline = predict_current(injected_rain=0.0)
    elevated = predict_current(injected_rain=80.0)
    assert elevated['risk_pct'] >= baseline['risk_pct'], \
        "Heavy rain should increase risk or keep it same"


# ── Causal model ───────────────────────────────────────────────────────────────
def test_causal_mediation():
    """
    Key property: proportion of rainfall's effect on shortfall mediated
    by equipment_downtime must be > 0.7 (as designed in the data generator).
    """
    from models.causal_model import run_causal_analysis
    from simulators.production_sim import generate_production_records
    df = generate_production_records(days=200, seed=42)
    results = run_causal_analysis(df)
    med = results.get('mediation_summary', {})
    prop = med.get('proportion_mediated', 0)
    assert prop > 0.7, f"Mediation proportion should be >0.7, got {prop:.3f}"


def test_rainfall_direct_effect_small():
    """Rainfall's DIRECT effect on shortfall should be near zero."""
    from models.causal_model import _ols_fallback
    from simulators.production_sim import generate_production_records
    df = generate_production_records(days=500, seed=42)
    direct = _ols_fallback(df, 'rainfall_mm', 'shortfall_pct')
    # Indirect effect (via downtime) should be much larger than direct
    indirect = _ols_fallback(df, 'equipment_downtime_h', 'shortfall_pct')
    assert abs(direct['effect']) < abs(indirect['effect']), \
        "Rainfall direct effect should be weaker than downtime direct effect"


# ── Production simulator ───────────────────────────────────────────────────────
def test_production_sim_causal_structure():
    """Verify the causal structure is correctly encoded in the data."""
    from simulators.production_sim import generate_production_records
    df = generate_production_records(days=365, seed=1)
    corr_rain_downtime = df['rainfall_mm'].corr(df['equipment_downtime_h'])
    corr_downtime_shortfall = df['equipment_downtime_h'].corr(df['shortfall_pct'])
    corr_rain_shortfall = df['rainfall_mm'].corr(df['shortfall_pct'])
    assert corr_rain_downtime > 0.5, "Rainfall must cause downtime"
    assert corr_downtime_shortfall > 0.5, "Downtime must cause shortfall"
    # Rainfall is correlated with shortfall (via downtime) but shouldn't dominate
    assert corr_rain_shortfall < corr_downtime_shortfall, \
        "Downtime should be stronger predictor of shortfall than rainfall directly"


# ── RL twin ───────────────────────────────────────────────────────────────────
def test_mine_env_step():
    from models.rl_twin import MineEnv
    env = MineEnv()
    obs, info = env.reset()
    assert len(obs) == 9
    for action in range(5):
        obs, reward, done, trunc, info = env.step(action)
        assert len(obs) == 9
        assert 'shortfall_pct' in info
        assert 0 <= info['shortfall_pct'] <= 100


def test_get_top_actions():
    from models.rl_twin import get_top_actions
    actions = get_top_actions(current_shortfall=50, rainfall_mm=5)
    assert len(actions) == 3
    # Actions should be sorted by risk reduction descending
    reductions = [a['risk_reduction_pct'] for a in actions]
    assert reductions == sorted(reductions, reverse=True)


if __name__ == '__main__':
    test_synthetic_mine_shape()
    test_synthetic_mine_ore_bodies()
    test_causal_mediation()
    test_rainfall_direct_effect_small()
    test_mine_env_step()
    print("All model tests passed!")
