-- Distribui as partidas unicas pelos formatos BO1, BO3 e BO5.
SELECT
    UPPER(best_of) AS formato,
    COUNT(*) AS partidas_unicas,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS participacao_pct
FROM fact_match
GROUP BY best_of
ORDER BY partidas_unicas DESC;
