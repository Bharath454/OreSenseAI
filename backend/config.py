"""
OreSense AI – Central Configuration
All settings loaded from environment variables with defaults.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Deployment mode ──────────────────────────────────────────────────────
    mode: str = "SIMULATED"   # SIMULATED | LIVE

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "postgresql://oresense:oresense_dev@localhost:5432/oresense"

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── MQTT ─────────────────────────────────────────────────────────────────
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883

    # ── Area of Interest (Balaghat manganese belt, MP, India) ────────────────
    aoi_minlat: float = 21.60
    aoi_maxlat: float = 21.73
    aoi_minlon: float = 80.58
    aoi_maxlon: float = 80.71
    grid_rows: int = 20
    grid_cols: int = 13

    # ── Satellite API credentials (optional – fallback to SIMULATED) ─────────
    copernicus_user: str = ""
    copernicus_password: str = ""
    sentinel_hub_client_id: str = ""
    sentinel_hub_client_secret: str = ""
    nasa_earthdata_user: str = ""
    nasa_earthdata_password: str = ""

    # ── Weather ──────────────────────────────────────────────────────────────
    open_meteo_url: str = "https://api.open-meteo.com/v1/forecast"

    # ── Model settings ───────────────────────────────────────────────────────
    model_retrain_interval_hours: int = 6
    fl_rounds: int = 5
    fl_min_clients: int = 3
    rl_timesteps: int = 50_000

    # ── Security ─────────────────────────────────────────────────────────────
    secret_key: str = "change_me_in_production_please"

    class Config:
        env_file = ".env"
        case_sensitive = False

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = {"SIMULATED", "LIVE"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v

    @property
    def is_live(self) -> bool:
        return self.mode == "LIVE"

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.grid_rows, self.grid_cols)

    @property
    def aoi_bounds(self) -> dict:
        return {
            "minlat": self.aoi_minlat,
            "maxlat": self.aoi_maxlat,
            "minlon": self.aoi_minlon,
            "maxlon": self.aoi_maxlon,
        }

    @property
    def aoi_center(self) -> tuple[float, float]:
        return (
            (self.aoi_minlat + self.aoi_maxlat) / 2,
            (self.aoi_minlon + self.aoi_maxlon) / 2,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
