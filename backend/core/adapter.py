"""
OreSense AI – Common Data Adapter Interface
============================================
Every data source (satellite, weather, subsurface) implements DataAdapter.
Two modes:
  LIVE      – real API call with retry/rate-limit handling
  SIMULATED – realistic synthetic generator derived from the hidden ore model

Each returned observation carries a DataMode tag so the UI can display
LIVE / CACHED / SIMULATED badges correctly.
"""
from __future__ import annotations

import abc
import enum
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


class DataMode(str, enum.Enum):
    LIVE = "LIVE"
    CACHED = "CACHED"
    SIMULATED = "SIMULATED"


@dataclass
class Observation:
    """A single data observation from any source."""
    source: str                         # e.g. "sentinel2", "open_meteo"
    mode: DataMode
    acquired_at: datetime               # when the data was captured/generated
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class DataAdapter(abc.ABC):
    """Base class for all data adapters."""

    source_name: str = "unknown"
    default_mode: DataMode = DataMode.SIMULATED

    def __init__(self, mode: DataMode | None = None):
        self.mode = mode or self.default_mode
        self._last_fetch: Optional[datetime] = None
        self._cache: Optional[Observation] = None

    @abc.abstractmethod
    async def fetch_live(self, **kwargs) -> Observation:
        """Fetch real data from the external API."""
        ...

    @abc.abstractmethod
    async def fetch_simulated(self, **kwargs) -> Observation:
        """Generate a realistic synthetic observation."""
        ...

    async def fetch(self, **kwargs) -> Observation:
        """
        Dispatch to the correct mode. If LIVE fetch fails, fall back to
        SIMULATED and mark the observation as SIMULATED with a warning.
        """
        if self.mode == DataMode.LIVE:
            try:
                obs = await self.fetch_live(**kwargs)
                obs.mode = DataMode.LIVE
                self._cache = obs
                self._last_fetch = datetime.now(timezone.utc)
                logger.info("adapter.live_ok", source=self.source_name)
                return obs
            except Exception as exc:
                logger.warning(
                    "adapter.live_failed_fallback",
                    source=self.source_name,
                    error=str(exc),
                )
                # Return cached if available, else simulate
                if self._cache is not None:
                    cached = self._cache
                    cached.mode = DataMode.CACHED
                    cached.metadata["fallback_reason"] = str(exc)
                    return cached
        # SIMULATED path
        obs = await self.fetch_simulated(**kwargs)
        obs.mode = DataMode.SIMULATED
        return obs

    def status(self) -> Dict[str, Any]:
        return {
            "source": self.source_name,
            "mode": self.mode.value,
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
            "has_cache": self._cache is not None,
        }
