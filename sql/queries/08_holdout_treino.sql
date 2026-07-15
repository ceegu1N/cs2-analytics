-- Compara o perfil contextual dos recortes de treino e holdout.
SELECT
    season_usage AS recorte,
    COUNT(*) AS partidas,
    ROUND(AVG(ABS(diff_elo_pre_match)), 2) AS diferenca_elo_media,
    ROUND(100.0 * AVG(context_is_lan), 2) AS partidas_lan_pct,
    SUM(CASE WHEN context_is_bo1 = 1 THEN 1 ELSE 0 END) AS series_bo1,
    SUM(CASE WHEN context_is_bo3 = 1 THEN 1 ELSE 0 END) AS series_bo3,
    SUM(CASE WHEN context_is_bo5 = 1 THEN 1 ELSE 0 END) AS series_bo5,
    ROUND(AVG(win_target), 4) AS taxa_vitoria_equipe_referencia
FROM fact_model_match
GROUP BY season_usage
ORDER BY CASE season_usage WHEN 'train' THEN 1 ELSE 2 END;
