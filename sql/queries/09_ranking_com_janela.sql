-- Ranqueia mensalmente as cinco equipes com maior taxa de vitoria.
WITH monthly_team AS (
    SELECT
        d.year_month AS mes,
        t.team_name AS equipe,
        COUNT(*) AS partidas,
        SUM(f.win) AS vitorias,
        AVG(f.win) AS taxa_vitoria
    FROM fact_team_match AS f
    JOIN dim_date AS d ON d.date_key = f.match_date_key
    JOIN dim_team AS t ON t.team_id = f.team_id
    GROUP BY d.year_month, t.team_id, t.team_name
    HAVING COUNT(*) >= 3
),
ranked AS (
    SELECT
        mes,
        equipe,
        partidas,
        vitorias,
        ROUND(taxa_vitoria, 4) AS taxa_vitoria,
        ROW_NUMBER() OVER (
            PARTITION BY mes
            ORDER BY taxa_vitoria DESC, partidas DESC, equipe
        ) AS posicao_no_mes
    FROM monthly_team
)
SELECT mes, posicao_no_mes, equipe, partidas, vitorias, taxa_vitoria
FROM ranked
WHERE posicao_no_mes <= 5
ORDER BY mes, posicao_no_mes;
