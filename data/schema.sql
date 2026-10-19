-- Схема БД проекта. Выполняется автоматически при первом старте контейнера postgres
-- (см. docker-compose.yml), либо вручную: psql -f data/schema.sql

-- ======================= Посадочная зона (raw): данные API «как есть» =======================
-- БЕЗ ограничений уникальности: реальная выгрузка содержит дубликаты (≈9% строк в наборе метро),
-- принимаем её как есть и чистим на следующем шаге.
CREATE TABLE IF NOT EXISTS metro_quarterly_raw (
    global_id     BIGINT,
    station       TEXT    NOT NULL,
    line          TEXT    NOT NULL,
    year          INT     NOT NULL,
    quarter_name  TEXT    NOT NULL,          -- 'I квартал' ... 'IV квартал'
    incoming      BIGINT,                    -- входы через турникеты
    outgoing      BIGINT                     -- выходы
);
CREATE TABLE IF NOT EXISTS transport_monthly_raw (
    year          INT  NOT NULL,
    month_name    TEXT NOT NULL,
    transport     TEXT NOT NULL,
    passengers    BIGINT
);
CREATE TABLE IF NOT EXISTS transport_yearly_raw (
    year          INT  NOT NULL,
    transport     TEXT NOT NULL,
    passengers    BIGINT
);

-- ======================= Чистый слой: уникальность гарантирует первичный ключ =======================
CREATE TABLE IF NOT EXISTS metro_quarterly_clean (
    station       TEXT    NOT NULL,
    line          TEXT    NOT NULL,
    year          INT     NOT NULL,
    quarter       SMALLINT NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    incoming      BIGINT  CHECK (incoming >= 0),
    outgoing      BIGINT  CHECK (outgoing >= 0),
    is_closed     BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (station, line, year, quarter)
);

-- ======================= Результаты моделей =======================
CREATE TABLE IF NOT EXISTS forecasts (
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    station       TEXT NOT NULL,
    line          TEXT NOT NULL,
    period        TEXT NOT NULL,             -- '2026Q3'
    forecast      BIGINT NOT NULL,
    low           BIGINT,
    high          BIGINT,
    model_version TEXT
);

CREATE TABLE IF NOT EXISTS detected_anomalies (
    detected_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    station       TEXT NOT NULL,
    line          TEXT NOT NULL,
    period        TEXT NOT NULL,
    actual        DOUBLE PRECISION,
    expected      DOUBLE PRECISION,
    anomaly_type  TEXT NOT NULL,             -- closure / partial_closure / station_drop / station_surge / line_event / imbalance
    severity      TEXT NOT NULL,             -- warning / critical
    explanation   TEXT,
    model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_anomalies_period ON detected_anomalies (period, severity);

CREATE TABLE IF NOT EXISTS data_quality_reports (
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    report        JSONB NOT NULL
);

-- ======================= Аналитические витрины (SQL-навык из вакансии) =======================
-- Годовая динамика станции: оконная функция LAG по тому же кварталу прошлого года
CREATE OR REPLACE VIEW v_station_yoy AS
SELECT station, line, year, quarter, incoming,
       LAG(incoming) OVER (PARTITION BY station, line, quarter ORDER BY year)          AS incoming_last_year,
       ROUND(100.0 * (incoming::numeric / NULLIF(LAG(incoming) OVER (
           PARTITION BY station, line, quarter ORDER BY year), 0) - 1), 1)            AS yoy_pct
FROM metro_quarterly_clean
WHERE NOT is_closed;

-- Покрытие квартала: сумма по станциям vs месячный итог метро (та же сверка, что в src/quality.py)
CREATE OR REPLACE VIEW v_quarter_coverage AS
WITH st AS (
    SELECT year, quarter, COUNT(*) AS n_stations, SUM(incoming) AS stations_sum
    FROM metro_quarterly_clean
    WHERE line NOT IN ('Московское центральное кольцо', 'Московская монорельсовая транспортная система')
    GROUP BY year, quarter
), mm AS (
    SELECT year,
           (ARRAY[1,1,1,2,2,2,3,3,3,4,4,4])[
               ARRAY_POSITION(ARRAY['Январь','Февраль','Март','Апрель','Май','Июнь','Июль','Август',
                                    'Сентябрь','Октябрь','Ноябрь','Декабрь'], month_name)] AS quarter,
           SUM(passengers) AS monthly_sum
    FROM transport_monthly_raw
    WHERE transport = 'Московский метрополитен'
    GROUP BY 1, 2
)
SELECT st.year, st.quarter, st.n_stations, st.stations_sum, mm.monthly_sum,
       ROUND(st.stations_sum::numeric / NULLIF(mm.monthly_sum, 0), 3) AS coverage
FROM st LEFT JOIN mm USING (year, quarter)
ORDER BY st.year, st.quarter;
