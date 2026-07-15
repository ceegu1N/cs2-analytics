-- Contrasta equilibrio pre-jogo e resultados entre partidas LAN e online.
WITH context_data AS (
    SELECT
        CASE WHEN context_is_lan = 1 THEN 'LAN' ELSE 'ONLINE' END AS ambiente,
        ABS(diff_elo_pre_match) AS diferenca_elo_absoluta,
        CASE
            WHEN diff_elo_pre_match > 0 AND win_target = 1 THEN 1.0
            WHEN diff_elo_pre_match < 0 AND win_target = 0 THEN 1.0
            WHEN diff_elo_pre_match <> 0 THEN 0.0
        END AS maior_elo_venceu
    FROM fact_model_match
)
SELECT
    ambiente,
    COUNT(*) AS partidas,
    ROUND(AVG(diferenca_elo_absoluta), 2) AS diferenca_elo_media,
    ROUND(100.0 * AVG(CASE WHEN diferenca_elo_absoluta < 50 THEN 1.0 ELSE 0.0 END), 2)
        AS confrontos_equilibrados_pct,
    ROUND(AVG(maior_elo_venceu), 4) AS taxa_vitoria_maior_elo
FROM context_data
GROUP BY ambiente
ORDER BY ambiente;
