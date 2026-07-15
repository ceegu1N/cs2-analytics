-- Resume os confrontos diretos mais recorrentes entre pares de equipes.
SELECT
    ta.team_name AS equipe_a,
    tb.team_name AS equipe_b,
    COUNT(*) AS confrontos,
    SUM(f.team_a_win) AS vitorias_equipe_a,
    COUNT(*) - SUM(f.team_a_win) AS vitorias_equipe_b,
    ROUND(AVG(f.team_a_win), 4) AS taxa_vitoria_equipe_a
FROM fact_match AS f
JOIN dim_team AS ta ON ta.team_id = f.team_a_id
JOIN dim_team AS tb ON tb.team_id = f.team_b_id
GROUP BY ta.team_id, ta.team_name, tb.team_id, tb.team_name
HAVING COUNT(*) >= 2
ORDER BY confrontos DESC, equipe_a, equipe_b
LIMIT 30;
