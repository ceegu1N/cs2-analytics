-- Compara o desempenho das equipes com pelo menos 50 perspectivas registradas.
SELECT
    t.team_name AS equipe,
    COUNT(*) AS partidas_representadas,
    SUM(f.win) AS vitorias,
    COUNT(*) - SUM(f.win) AS derrotas,
    ROUND(AVG(f.win), 4) AS taxa_vitoria
FROM fact_team_match AS f
JOIN dim_team AS t ON t.team_id = f.team_id
GROUP BY t.team_id, t.team_name
HAVING COUNT(*) >= 50
ORDER BY taxa_vitoria DESC, partidas_representadas DESC, equipe;
