"""
OreSense AI – Weather Adapter (Open-Meteo)
==========================================
LIVE mode:  Open-Meteo free API – no API key required.
            Fetches hourly temperature and precipitation for the AOI centroid.
            Derives a land-surface temperature (LST) anomaly vs the seasonal mean.
SIMULATED:  Seasonal monsoon pattern with rainfall events correlated to
            shortfall risk (used by the causal model).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
import numpy as np

from core.adapter import DataAdapter, DataMode, Observation
from config import settings


# 30-year monthly normals for Balaghat area (°C and mm/month, approximate)
TEMP_NORMALS_C = [22, 25, 30, 36, 38, 34, 28, 27, 28, 28, 24, 21]
RAIN_NORMALS_MM_MONTH = [10, 8, 12, 15, 25, 180, 310, 280, 180, 60, 15, 8]


def _monthly_normal_rain_mm_day(month: int) -> float:
    return RAIN_NORMALS_MM_MONTH[month - 1] / 30.0


def _monthly_normal_temp_c(month: int) -> float:
    return TEMP_NORMALS_C[month - 1]


class WeatherAdapter(DataAdapter):
    source_name = "open_meteo"

    OPEN_METEO_URL = settings.open_meteo_url
    lat, lon = settings.aoi_center

    async def fetch_live(self, **kwargs) -> Observation:
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": "temperature_2m,precipitation,soil_moisture_0_1cm",
            "forecast_days": 1,
            "timezone": "Asia/Kolkata",
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(self.OPEN_METEO_URL, params=params)
            resp.raise_for_status()
            raw = resp.json()

        hourly = raw.get("hourly", {})
        now_h = datetime.now().hour
        temp_c = hourly.get("temperature_2m", [None] * 24)[now_h] or 30.0
        precip_mm = hourly.get("precipitation", [0.0] * 24)[now_h] or 0.0
        sm = hourly.get("soil_moisture_0_1cm", [0.3] * 24)[now_h] or 0.3

        month = datetime.now().month
        lst_anomaly = temp_c - _monthly_normal_temp_c(month)
        rain_anomaly = precip_mm - _monthly_normal_rain_mm_day(month)

        return Observation(
            source=self.source_name,
            mode=DataMode.LIVE,
            acquired_at=datetime.now(timezone.utc),
            data={
                "temperature_c": round(temp_c, 2),
                "rainfall_mm": round(precip_mm, 3),
                "soil_moisture_surface": round(float(sm), 3),
                "lst_anomaly": round(lst_anomaly, 2),
                "rain_anomaly_mm": round(rain_anomaly, 3),
            },
            metadata={"api": "open-meteo.com", "lat": self.lat, "lon": self.lon},
        )

    async def fetch_simulated(self, **kwargs) -> Observation:
        """
        Monsoon-realistic weather simulation for Balaghat.
        Rainfall is the key causal driver for equipment downtime and shortfall.
        Active simulator events (heavy rain, etc.) are applied here.
        """
        rng = np.random.default_rng(int(datetime.utcnow().timestamp() / 3600) % 100000)
        month = datetime.utcnow().month
        doy = datetime.utcnow().timetuple().tm_yday

        # Base rainfall using seasonal distribution
        base_rain_mm = _monthly_normal_rain_mm_day(month)
        # Occasional heavy rain events
        heavy = float(rng.exponential(base_rain_mm * 2)) if rng.random() < 0.25 else 0.0
        rainfall_mm = max(0.0, float(rng.normal(base_rain_mm, base_rain_mm * 0.4)) + heavy)

        # Check for injected heavy-rain events (from simulator control panel)
        injected = kwargs.get("injected_rain_mm", 0.0)
        rainfall_mm = max(rainfall_mm, injected)

        # Temperature
        temp_c = _monthly_normal_temp_c(month) + float(rng.normal(0, 2.5))
        lst_anomaly = temp_c - _monthly_normal_temp_c(month)

        # Soil moisture proxy
        sm_surface = min(0.05 + 0.08 * rainfall_mm / 20.0 + float(rng.normal(0, 0.02)), 0.6)

        return Observation(
            source=self.source_name,
            mode=DataMode.SIMULATED,
            acquired_at=datetime.now(timezone.utc),
            data={
                "temperature_c": round(temp_c, 2),
                "rainfall_mm": round(rainfall_mm, 3),
                "soil_moisture_surface": round(sm_surface, 3),
                "lst_anomaly": round(lst_anomaly, 2),
                "rain_anomaly_mm": round(rainfall_mm - base_rain_mm, 3),
            },
            metadata={"note": "SIMULATED – seasonal monsoon model for Balaghat MP"},
        )
