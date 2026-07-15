-- Resume volume, periodo e granularidade das bases operacionais e de modelagem.
SELECT
    (SELECT COUNT(*) FROM fact_match) AS partidas_unicas,
    (SELECT COUNT(*) FROM fact_team_match) AS perspectivas_de_equipe,
    (SELECT COUNT(*) FROM fact_model_match) AS partidas_modelagem,
    (SELECT COUNT(DISTINCT team_id) FROM fact_team_match) AS equipes_representadas,
    (SELECT COUNT(*) FROM dim_team) AS equipes_catalogadas,
    (SELECT MIN(date_iso) FROM dim_date) AS primeira_data,
    (SELECT MAX(date_iso) FROM dim_date) AS ultima_data,
    (SELECT COUNT(*) FROM fact_match WHERE lan_online = 'LAN') AS partidas_lan,
    (SELECT COUNT(*) FROM fact_match WHERE lan_online = 'ONLINE') AS partidas_online;
