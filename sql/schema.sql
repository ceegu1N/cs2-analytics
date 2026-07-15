PRAGMA foreign_keys = ON;

CREATE TABLE dim_team (
    team_id INTEGER PRIMARY KEY,
    team_key TEXT NOT NULL UNIQUE,
    team_name TEXT NOT NULL,
    hltv_team_id INTEGER
);

CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,
    date_iso TEXT NOT NULL UNIQUE,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    semester INTEGER NOT NULL,
    year_month TEXT NOT NULL
);

CREATE TABLE fact_team_match (
    perspective_id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL,
    match_date_key INTEGER NOT NULL,
    season_label TEXT NOT NULL,
    season_usage TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    opponent_team_id INTEGER NOT NULL,
    team_score INTEGER NOT NULL,
    opponent_score INTEGER NOT NULL,
    win INTEGER NOT NULL CHECK (win IN (0, 1)),
    best_of TEXT NOT NULL CHECK (best_of IN ('bo1', 'bo3', 'bo5')),
    lan_online TEXT NOT NULL CHECK (lan_online IN ('LAN', 'ONLINE')),
    match_url TEXT NOT NULL,
    UNIQUE (match_id, team_id),
    FOREIGN KEY (match_date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY (team_id) REFERENCES dim_team(team_id),
    FOREIGN KEY (opponent_team_id) REFERENCES dim_team(team_id)
);

CREATE TABLE fact_match (
    match_id INTEGER PRIMARY KEY,
    match_date_key INTEGER NOT NULL,
    season_label TEXT NOT NULL,
    season_usage TEXT NOT NULL,
    team_a_id INTEGER NOT NULL,
    team_b_id INTEGER NOT NULL,
    team_a_score INTEGER NOT NULL,
    team_b_score INTEGER NOT NULL,
    team_a_win INTEGER NOT NULL CHECK (team_a_win IN (0, 1)),
    best_of TEXT NOT NULL CHECK (best_of IN ('bo1', 'bo3', 'bo5')),
    lan_online TEXT NOT NULL CHECK (lan_online IN ('LAN', 'ONLINE')),
    match_url TEXT NOT NULL,
    FOREIGN KEY (match_date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY (team_a_id) REFERENCES dim_team(team_id),
    FOREIGN KEY (team_b_id) REFERENCES dim_team(team_id)
);

CREATE TABLE fact_model_match (
    model_match_id INTEGER PRIMARY KEY,
    match_date_key INTEGER NOT NULL,
    season_label TEXT NOT NULL,
    season_usage TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    opponent_team_id INTEGER NOT NULL,
    win_target INTEGER NOT NULL CHECK (win_target IN (0, 1)),
    diff_recent_win_rate REAL NOT NULL,
    diff_maps_played_mean_5 REAL NOT NULL,
    diff_rounds_played_mean_5 REAL NOT NULL,
    diff_total_kills_mean_5 REAL NOT NULL,
    diff_rating_mean_5 REAL NOT NULL,
    diff_adr_mean_5 REAL NOT NULL,
    diff_impact_mean_5 REAL NOT NULL,
    diff_kast_mean_5 REAL NOT NULL,
    diff_kd_mean_5 REAL NOT NULL,
    diff_kills_per_round_mean_5 REAL NOT NULL,
    diff_deaths_per_round_mean_5 REAL NOT NULL,
    diff_assists_per_round_mean_5 REAL NOT NULL,
    diff_headshot_pct_mean_5 REAL NOT NULL,
    diff_saved_by_teammate_per_round_mean_5 REAL NOT NULL,
    diff_utility_damage_per_round_mean_5 REAL NOT NULL,
    diff_flash_assists_per_round_mean_5 REAL NOT NULL,
    diff_utility_kills_per_100_rounds_mean_5 REAL NOT NULL,
    diff_time_opponent_flashed_per_round_mean_5 REAL NOT NULL,
    diff_opening_kills_per_round_mean_5 REAL NOT NULL,
    diff_opening_deaths_per_round_mean_5 REAL NOT NULL,
    diff_opening_success_mean_5 REAL NOT NULL,
    diff_win_after_opening_kill_pct_mean_5 REAL NOT NULL,
    diff_trade_kills_per_round_mean_5 REAL NOT NULL,
    diff_saved_teammate_per_round_mean_5 REAL NOT NULL,
    diff_last_alive_pct_mean_5 REAL NOT NULL,
    diff_one_on_one_win_pct_mean_5 REAL NOT NULL,
    diff_time_alive_per_round_mean_5 REAL NOT NULL,
    diff_rounds_with_a_kill_mean_5 REAL NOT NULL,
    diff_kills_per_round_win_mean_5 REAL NOT NULL,
    diff_damage_per_round_win_mean_5 REAL NOT NULL,
    diff_pistol_round_rating_mean_5 REAL NOT NULL,
    diff_players_count REAL NOT NULL,
    diff_missing_players REAL NOT NULL,
    context_is_lan REAL NOT NULL,
    context_is_bo1 REAL NOT NULL,
    context_is_bo3 REAL NOT NULL,
    context_is_bo5 REAL NOT NULL,
    diff_elo_pre_match REAL NOT NULL,
    FOREIGN KEY (match_date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY (team_id) REFERENCES dim_team(team_id),
    FOREIGN KEY (opponent_team_id) REFERENCES dim_team(team_id)
);

CREATE TABLE fact_prediction_holdout (
    model_match_id INTEGER PRIMARY KEY,
    match_date_key INTEGER NOT NULL,
    match_date_iso TEXT NOT NULL,
    season_label TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    team_name TEXT NOT NULL,
    opponent_team_id INTEGER NOT NULL,
    opponent_name TEXT NOT NULL,
    environment TEXT NOT NULL CHECK (environment IN ('LAN', 'ONLINE')),
    abs_elo_difference REAL NOT NULL CHECK (abs_elo_difference >= 0),
    elo_band TEXT NOT NULL CHECK (elo_band IN ('0-49', '50-99', '100-199', '200+')),
    elo_band_sort INTEGER NOT NULL CHECK (elo_band_sort BETWEEN 1 AND 4),
    y_true INTEGER NOT NULL CHECK (y_true IN (0, 1)),
    actual_result TEXT NOT NULL CHECK (actual_result IN ('Vitoria', 'Derrota')),
    probability REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
    probability_band TEXT NOT NULL,
    probability_band_sort INTEGER NOT NULL CHECK (probability_band_sort BETWEEN 1 AND 10),
    predicted_class INTEGER NOT NULL CHECK (predicted_class IN (0, 1)),
    predicted_result TEXT NOT NULL CHECK (predicted_result IN ('Vitoria', 'Derrota')),
    correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.5 AND 1),
    high_confidence INTEGER NOT NULL CHECK (high_confidence IN (0, 1)),
    high_confidence_error INTEGER NOT NULL CHECK (high_confidence_error IN (0, 1)),
    confusion_cell TEXT NOT NULL,
    brier_component REAL NOT NULL CHECK (brier_component BETWEEN 0 AND 1),
    log_loss_component REAL NOT NULL CHECK (log_loss_component >= 0),
    FOREIGN KEY (model_match_id) REFERENCES fact_model_match(model_match_id),
    FOREIGN KEY (match_date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY (team_id) REFERENCES dim_team(team_id),
    FOREIGN KEY (opponent_team_id) REFERENCES dim_team(team_id)
);

CREATE TABLE build_metadata (
    metadata_key TEXT PRIMARY KEY,
    metadata_value TEXT NOT NULL
);

CREATE INDEX idx_team_match_team ON fact_team_match(team_id);
CREATE INDEX idx_team_match_date ON fact_team_match(match_date_key);
CREATE INDEX idx_match_date ON fact_match(match_date_key);
CREATE INDEX idx_model_usage ON fact_model_match(season_usage);
CREATE INDEX idx_model_date ON fact_model_match(match_date_key);
CREATE INDEX idx_prediction_date ON fact_prediction_holdout(match_date_key);
CREATE INDEX idx_prediction_team ON fact_prediction_holdout(team_id);

CREATE VIEW vw_team_performance AS
SELECT
    t.team_id,
    t.team_name,
    COUNT(*) AS represented_matches,
    SUM(f.win) AS wins,
    COUNT(*) - SUM(f.win) AS losses,
    ROUND(AVG(f.win), 4) AS win_rate
FROM fact_team_match f
JOIN dim_team t ON t.team_id = f.team_id
GROUP BY t.team_id, t.team_name;

CREATE VIEW vw_model_context AS
SELECT
    CASE WHEN context_is_lan = 1 THEN 'LAN' ELSE 'ONLINE' END AS environment,
    CASE
        WHEN context_is_bo1 = 1 THEN 'BO1'
        WHEN context_is_bo3 = 1 THEN 'BO3'
        WHEN context_is_bo5 = 1 THEN 'BO5'
        ELSE 'OUTRO'
    END AS series_format,
    season_usage,
    COUNT(*) AS matches,
    ROUND(AVG(win_target), 4) AS reference_team_win_rate
FROM fact_model_match
GROUP BY environment, series_format, season_usage;
