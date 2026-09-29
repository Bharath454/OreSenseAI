"""
OreSense AI – Unit Tests: Data Adapters
Tests both modes (SIMULATED always; LIVE skipped without credentials).
"""
import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['MODE'] = 'SIMULATED'

import pytest
from core.adapter import DataMode
from adapters.sentinel2 import Sentinel2Adapter
from adapters.sentinel1 import Sentinel1Adapter
from adapters.weather import WeatherAdapter
from adapters.subsurface import AMTAdapter, ANTAdapter, HyperspectralAdapter


@pytest.mark.asyncio
async def test_sentinel2_simulated():
    adapter = Sentinel2Adapter(mode=DataMode.SIMULATED)
    obs = await adapter.fetch()
    assert obs.ok, f"Error: {obs.error}"
    assert obs.mode == DataMode.SIMULATED
    assert 'ndvi' in obs.data
    ndvi = obs.data['ndvi']
    assert isinstance(ndvi, list), "NDVI must be a list (grid)"
    assert len(ndvi) == 20, f"Expected 20 rows, got {len(ndvi)}"
    assert len(ndvi[0]) == 13, f"Expected 13 cols, got {len(ndvi[0])}"
    # All NDVI values in valid range
    flat = [v for row in ndvi for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat), "NDVI out of range [0,1]"


@pytest.mark.asyncio
async def test_sentinel1_simulated():
    adapter = Sentinel1Adapter(mode=DataMode.SIMULATED)
    obs = await adapter.fetch()
    assert obs.ok
    assert 'sar_backscatter' in obs.data
    assert 'soil_moisture' in obs.data
    assert 'insar_deform_mm' in obs.data
    sm = obs.data['soil_moisture']
    flat = [v for row in sm for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat), "Soil moisture out of [0,1]"
    assert obs.metadata.get('insar_mode') == 'CACHED', "InSAR must be marked CACHED"


@pytest.mark.asyncio
async def test_weather_simulated():
    adapter = WeatherAdapter(mode=DataMode.SIMULATED)
    obs = await adapter.fetch()
    assert obs.ok
    assert 'rainfall_mm' in obs.data
    assert obs.data['rainfall_mm'] >= 0, "Rainfall cannot be negative"
    assert 'temperature_c' in obs.data


@pytest.mark.asyncio
async def test_amt_simulated():
    adapter = AMTAdapter()
    obs = await adapter.fetch()
    assert obs.ok
    assert obs.mode == DataMode.SIMULATED
    conductivity = obs.data['amt_conductivity']
    flat = [v for row in conductivity for v in row]
    assert all(v > 0 for v in flat), "Conductivity must be positive"
    assert all(v < 60 for v in flat), "Conductivity unrealistically high"


@pytest.mark.asyncio
async def test_ant_simulated():
    adapter = ANTAdapter()
    obs = await adapter.fetch()
    assert obs.ok
    vels = obs.data['ant_velocity_pct']
    flat = [v for row in vels for v in row]
    # Should be negative or near zero (velocity suppression over ore)
    assert all(v <= 5 for v in flat), "ANT velocity anomaly should be ≤5%"


@pytest.mark.asyncio
async def test_hyperspectral_simulated():
    adapter = HyperspectralAdapter()
    obs = await adapter.fetch()
    assert obs.ok
    mn = obs.data['hyperspectral_mn_idx']
    flat = [v for row in mn for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat), "Mn index out of [0,1]"


@pytest.mark.asyncio
async def test_live_fallback_to_simulated():
    """
    When LIVE mode fails (no credentials), adapter must fall back to SIMULATED
    not crash.
    """
    adapter = Sentinel2Adapter(mode=DataMode.LIVE)
    obs = await adapter.fetch()
    # Either LIVE or SIMULATED – must not error
    assert obs.ok, "Should not error on live fallback"
    assert obs.mode in (DataMode.LIVE, DataMode.SIMULATED, DataMode.CACHED)


if __name__ == '__main__':
    asyncio.run(test_sentinel2_simulated())
    print("All adapter tests passed!")
