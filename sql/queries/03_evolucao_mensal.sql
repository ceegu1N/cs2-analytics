-- Agrega o volume mensal de partidas unicas por ambiente competitivo.
SELECT
    d.year_month AS mes,
    COUNT(*) AS partidas_unicas,
    SUM(CASE WHEN f.lan_online = 'LAN' THEN 1 ELSE 0 END) AS partidas_lan,
    SUM(CASE WHEN f.lan_online = 'ONLINE' THEN 1 ELSE 0 END) AS partidas_online,
    SUM(CASE WHEN f.best_of = 'bo1' THEN 1 ELSE 0 END) AS series_bo1,
    SUM(CASE WHEN f.best_of = 'bo3' THEN 1 ELSE 0 END) AS series_bo3,
    SUM(CASE WHEN f.best_of = 'bo5' THEN 1 ELSE 0 END) AS series_bo5
FROM fact_match AS f
JOIN dim_date AS d ON d.date_key = f.match_date_key
GROUP BY d.year_month
ORDER BY d.year_month;
