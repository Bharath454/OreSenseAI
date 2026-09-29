-- OreSense AI – PostgreSQL/PostGIS/TimescaleDB initialisation
-- Runs once on first container start.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─── Grid cells (spatial) ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS grid_cells (
    id          SERIAL PRIMARY KEY,
    row_idx     INTEGER NOT NULL,
    col_idx     INTEGER NOT NULL,
    geom        GEOMETRY(Polygon, 4326) NOT NULL,
    centroid    GEOMETRY(Point,   4326) NOT NULL,
    elevation_m REAL DEFAULT 0,
    UNIQUE (row_idx, col_idx)
);
CREATE INDEX IF NOT EXISTS idx_grid_geom ON grid_cells USING GIST (geom);

-- ─── Remote-sensing observations (time-series) ───────────────────────────────
CREATE TABLE IF NOT EXISTS rs_observations (
    time            TIMESTAMPTZ NOT NULL,
    cell_id         INTEGER     NOT NULL REFERENCES grid_cells(id),
    source          TEXT        NOT NULL,   -- sentinel2, landsat8, etc.
    data_mode       TEXT        NOT NULL,   -- LIVE | CACHED | SIMULATED
    ndvi            REAL,
    ndvi_anomaly    REAL,
    sar_backscatter REAL,
    soil_moisture   REAL,
    lst_anomaly     REAL,
    hyperspectral_mn_idx REAL,
    rainfall_mm     REAL,
    amt_conductivity REAL,
    ant_velocity_pct REAL,
    insar_deform_mm  REAL
);
SELECT create_hypertable('rs_observations','time', if_not_exists => TRUE);

-- ─── Reserve estimates ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reserve_estimates (
    time            TIMESTAMPTZ NOT NULL,
    cell_id         INTEGER     NOT NULL REFERENCES grid_cells(id),
    model_version   TEXT        NOT NULL,
    ore_prob        REAL,       -- 0–1
    ore_uncertainty REAL,       -- std dev across trees
    reserve_class   TEXT,       -- Measured / Indicated / Inferred / Below
    feature_json    JSONB
);
SELECT create_hypertable('reserve_estimates','time', if_not_exists => TRUE);

-- ─── Production records ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS production_records (
    date            DATE        PRIMARY KEY,
    site            TEXT        NOT NULL DEFAULT 'Balaghat',
    planned_tonnes  REAL,
    actual_tonnes   REAL,
    shortfall_pct   REAL,
    rainfall_mm     REAL,
    equipment_downtime_h REAL,
    crew_available  INTEGER,
    blasting_delays_h REAL
);

-- ─── Shortfall predictions ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS shortfall_predictions (
    time            TIMESTAMPTZ NOT NULL,
    model_version   TEXT        NOT NULL,
    risk_pct        REAL,
    uncertainty     REAL,
    top_drivers     JSONB,
    forecast_json   JSONB       -- 30-day daily risk array
);
SELECT create_hypertable('shortfall_predictions','time', if_not_exists => TRUE);

-- ─── Causal estimates ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS causal_estimates (
    id              SERIAL PRIMARY KEY,
    computed_at     TIMESTAMPTZ DEFAULT NOW(),
    model_version   TEXT,
    results_json    JSONB
);

-- ─── IoT telemetry ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS iot_telemetry (
    time            TIMESTAMPTZ NOT NULL,
    machine_id      TEXT        NOT NULL,
    lat             REAL,
    lon             REAL,
    vibration_g     REAL,
    fuel_rate_lph   REAL,
    state           TEXT,       -- OPERATING | IDLE | BREAKDOWN | MAINTENANCE
    health_score    REAL
);
SELECT create_hypertable('iot_telemetry','time', if_not_exists => TRUE);

-- ─── Model registry ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL PRIMARY KEY,
    model_name      TEXT        NOT NULL,
    version         TEXT        NOT NULL,
    trained_at      TIMESTAMPTZ DEFAULT NOW(),
    metrics_json    JSONB,
    is_active       BOOLEAN     DEFAULT TRUE
);

-- ─── Federated learning rounds ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fl_rounds (
    id              SERIAL PRIMARY KEY,
    round_num       INTEGER,
    completed_at    TIMESTAMPTZ DEFAULT NOW(),
    global_accuracy REAL,
    client_metrics  JSONB
);

-- ─── Events / simulator injections ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS simulator_events (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    event_type      TEXT,       -- heavy_rain | equipment_breakdown | crew_shortage
    parameters      JSONB,
    active_until    TIMESTAMPTZ
);
