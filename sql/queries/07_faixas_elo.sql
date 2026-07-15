-- Mede a associacao descritiva entre vantagem de ELO pre-jogo e vitoria.
WITH elo_matches AS (
    SELECT
        ABS(diff_elo_pre_match) AS diferenca_elo_absoluta,
        CASE
            WHEN diff_elo_pre_match > 0 AND win_target = 1 THEN 1.0
            WHEN diff_elo_pre_match < 0 AND win_target = 0 THEN 1.0
            WHEN diff_elo_pre_match <> 0 THEN 0.0
        END AS maior_elo_venceu
    FROM fact_model_match
),
elo_bands AS (
    SELECT
        CASE
            WHEN diferenca_elo_absoluta < 50 THEN '0-49'
            WHEN diferenca_elo_absoluta < 100 THEN '50-99'
            WHEN diferenca_elo_absoluta < 200 THEN '100-199'
            ELSE '200+'
        END AS faixa_elo,
        CASE
            WHEN diferenca_elo_absoluta < 50 THEN 1
            WHEN diferenca_elo_absoluta < 100 THEN 2
            WHEN diferenca_elo_absoluta < 200 THEN 3
            ELSE 4
        END AS ordem_faixa,
        diferenca_elo_absoluta,
        maior_elo_venceu
    FROM elo_matches
)
SELECT
    faixa_elo,
    COUNT(*) AS partidas,
    ROUND(AVG(diferenca_elo_absoluta), 2) AS diferenca_elo_media,
    ROUND(AVG(maior_elo_venceu), 4) AS taxa_vitoria_maior_elo
FROM elo_bands
GROUP BY faixa_elo, ordem_faixa
ORDER BY ordem_faixa;
