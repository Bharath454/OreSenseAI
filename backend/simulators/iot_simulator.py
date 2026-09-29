"""
OreSense AI – Equipment IoT Simulator
======================================
Simulates 8 mining machines (excavators, dumpers, drills) publishing
telemetry every 2 seconds over MQTT.

MQTT Topics:
  oresense/iot/{machine_id}/telemetry  → JSON payload
  oresense/iot/{machine_id}/alert      → breakdown / maintenance alerts

Each machine has:
  - GPS position (wanders within AOI)
  - Vibration (g-force, higher = possible fault)
  - Fuel consumption rate (L/h)
  - Operating state: OPERATING | IDLE | BREAKDOWN | MAINTENANCE

Breakdown model:
  - Base failure probability: 0.002/cycle
  - Increases with vibration, poor health score, rainfall
  - Weather-correlated downtime: heavy rain → IDLE
  - Recovery: BREAKDOWN → MAINTENANCE (4h) → OPERATING

Injected events from the Simulator Control Panel are read from
the DB 'simulator_events' table and applied each cycle.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import paho.mqtt.client as mqtt
import asyncpg
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://oresense:oresense_dev@localhost:5432/oresense"
)
AOI = dict(
    minlat=float(os.getenv("AOI_MINLAT", "21.60")),
    maxlat=float(os.getenv("AOI_MAXLAT", "21.73")),
    minlon=float(os.getenv("AOI_MINLON", "80.58")),
    maxlon=float(os.getenv("AOI_MAXLON", "80.71")),
)

MACHINES = [
    {"id": "EX-001", "type": "Excavator",    "base_vibration": 2.1},
    {"id": "EX-002", "type": "Excavator",    "base_vibration": 2.3},
    {"id": "DU-001", "type": "Dumper",       "base_vibration": 1.5},
    {"id": "DU-002", "type": "Dumper",       "base_vibration": 1.6},
    {"id": "DU-003", "type": "Dumper",       "base_vibration": 1.4},
    {"id": "DR-001", "type": "Drill",        "base_vibration": 3.2},
    {"id": "CR-001", "type": "Crusher",      "base_vibration": 4.1},
    {"id": "BL-001", "type": "Bulldozer",    "base_vibration": 2.0},
]

STATES = ["OPERATING", "IDLE", "BREAKDOWN", "MAINTENANCE"]
FUEL_RATE = {"Excavator": 35, "Dumper": 25, "Drill": 18, "Crusher": 45, "Bulldozer": 28}
RECOVERY_CYCLES = {"BREAKDOWN": 3600 // 2, "MAINTENANCE": 7200 // 2}  # cycles to recover


@dataclass
class MachineState:
    machine_id: str
    machine_type: str
    base_vibration: float
    lat: float
    lon: float
    state: str = "OPERATING"
    health_score: float = 1.0
    breakdown_countdown: int = 0
    total_cycles: int = 0

    def step(self, rain_mm: float, breakdown_injected: bool) -> Dict:
        """Advance one 2-second cycle and return telemetry dict."""
        self.total_cycles += 1
        rng = random.Random(int(time.time() * 1000) + hash(self.machine_id))

        # State transitions
        if self.state in ("BREAKDOWN", "MAINTENANCE"):
            self.breakdown_countdown -= 1
            if self.breakdown_countdown <= 0:
                self.state = "OPERATING" if self.state == "MAINTENANCE" else "MAINTENANCE"
                self.breakdown_countdown = RECOVERY_CYCLES.get(self.state, 0)
                self.health_score = min(1.0, self.health_score + 0.2)
        elif self.state == "IDLE":
            if rain_mm < 5 and rng.random() < 0.1:
                self.state = "OPERATING"
        else:  # OPERATING
            # Rain → IDLE
            if rain_mm > 20 or (rain_mm > 10 and rng.random() < 0.3):
                self.state = "IDLE"
            # Random / injected breakdown
            fail_prob = 0.002 * (2 - self.health_score)
            if breakdown_injected or rng.random() < fail_prob:
                self.state = "BREAKDOWN"
                self.breakdown_countdown = RECOVERY_CYCLES["BREAKDOWN"]
                self.health_score = max(0.1, self.health_score - 0.3)

        # Health degrades slowly while operating
        if self.state == "OPERATING":
            self.health_score = max(0.2, self.health_score - 0.0001)

        # GPS wander
        self.lat += rng.gauss(0, 0.0003)
        self.lon += rng.gauss(0, 0.0003)
        self.lat = max(AOI["minlat"], min(AOI["maxlat"], self.lat))
        self.lon = max(AOI["minlon"], min(AOI["maxlon"], self.lon))

        # Sensor readings
        vib = self.base_vibration * (1 + 0.3 * (1 - self.health_score))
        vib += rng.gauss(0, 0.2)
        if self.state != "OPERATING":
            vib = max(0.1, vib * 0.1)

        fuel = FUEL_RATE.get(self.machine_type, 25)
        if self.state == "OPERATING":
            fuel += rng.gauss(0, 3)
        else:
            fuel = rng.gauss(3, 0.5)

        return {
            "machine_id": self.machine_id,
            "machine_type": self.machine_type,
            "time": datetime.now(timezone.utc).isoformat(),
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "vibration_g": round(max(0.0, vib), 3),
            "fuel_rate_lph": round(max(0.0, fuel), 2),
            "state": self.state,
            "health_score": round(self.health_score, 3),
            "data_mode": "SIMULATED",
        }


async def get_injected_rain() -> float:
    """Check DB for active heavy-rain events."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        row = await conn.fetchrow(
            """
            SELECT parameters FROM simulator_events
            WHERE event_type = 'heavy_rain'
              AND active_until > NOW()
            ORDER BY created_at DESC LIMIT 1
            """
        )
        await conn.close()
        if row:
            params = json.loads(row["parameters"])
            return float(params.get("rain_mm", 0))
    except Exception:
        pass
    return 0.0


async def get_injected_breakdown() -> bool:
    """Check DB for active breakdown events."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        row = await conn.fetchrow(
            """
            SELECT id FROM simulator_events
            WHERE event_type = 'equipment_breakdown'
              AND active_until > NOW()
            ORDER BY created_at DESC LIMIT 1
            """
        )
        await conn.close()
        return row is not None
    except Exception:
        return False


async def insert_telemetry(records: List[Dict]) -> None:
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        await conn.executemany(
            """
            INSERT INTO iot_telemetry
            (time, machine_id, lat, lon, vibration_g, fuel_rate_lph, state, health_score)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            """,
            [
                (
                    r["time"],
                    r["machine_id"],
                    r["lat"],
                    r["lon"],
                    r["vibration_g"],
                    r["fuel_rate_lph"],
                    r["state"],
                    r["health_score"],
                )
                for r in records
            ],
        )
        await conn.close()
    except Exception as e:
        print(f"[IoT] DB insert failed: {e}")


def main():
    # Initialise machine states
    rng = random.Random(42)
    machines = []
    for spec in MACHINES:
        lat = rng.uniform(AOI["minlat"], AOI["maxlat"])
        lon = rng.uniform(AOI["minlon"], AOI["maxlon"])
        machines.append(
            MachineState(
                machine_id=spec["id"],
                machine_type=spec["type"],
                base_vibration=spec["base_vibration"],
                lat=lat,
                lon=lon,
            )
        )

    # MQTT client
    client = mqtt.Client(client_id="oresense_iot_simulator")
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_start()

    print(f"[IoT] Simulator started – {len(machines)} machines, MQTT {MQTT_BROKER}:{MQTT_PORT}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    cycle = 0
    while True:
        try:
            rain_mm = loop.run_until_complete(get_injected_rain())
            breakdown = loop.run_until_complete(get_injected_breakdown())
        except Exception:
            rain_mm, breakdown = 0.0, False

        telemetry_batch = []
        for m in machines:
            payload = m.step(rain_mm=rain_mm, breakdown_injected=breakdown)
            topic = f"oresense/iot/{m.machine_id}/telemetry"
            client.publish(topic, json.dumps(payload), qos=0)
            telemetry_batch.append(payload)

        # Write to DB every 10 cycles (every 20s) to avoid overloading
        if cycle % 10 == 0:
            loop.run_until_complete(insert_telemetry(telemetry_batch))

        cycle += 1
        time.sleep(2)


if __name__ == "__main__":
    main()
