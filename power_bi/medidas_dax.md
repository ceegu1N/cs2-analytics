# Catálogo de medidas DAX

DAX é a linguagem de medidas do Power BI. Ela não substitui SQL. As fórmulas abaixo pressupõem os nomes originais das tabelas exportadas.

## Indicadores globais

```DAX
Partidas Únicas = COUNTROWS(fact_match)
```

```DAX
Partidas LAN =
CALCULATE(
    [Partidas Únicas],
    fact_match[lan_online] = "LAN"
)
```

```DAX
Partidas Online =
CALCULATE(
    [Partidas Únicas],
    fact_match[lan_online] = "ONLINE"
)
```

```DAX
Participação LAN = DIVIDE([Partidas LAN], [Partidas Únicas])
```

## Desempenho por equipe

```DAX
Partidas Representadas = COUNTROWS(fact_team_match)
```

```DAX
Vitórias = SUM(fact_team_match[win])
```

```DAX
Derrotas = [Partidas Representadas] - [Vitórias]
```

```DAX
Taxa de Vitória = DIVIDE([Vitórias], [Partidas Representadas])
```

## Recorte de modelagem

```DAX
Partidas de Modelagem = COUNTROWS(fact_model_match)
```

```DAX
Diferença Média Absoluta de ELO =
AVERAGEX(
    fact_model_match,
    ABS(fact_model_match[diff_elo_pre_match])
)
```

```DAX
Vitória do Maior ELO =
AVERAGEX(
    FILTER(
        fact_model_match,
        fact_model_match[diff_elo_pre_match] <> 0
    ),
    VAR DiffElo = fact_model_match[diff_elo_pre_match]
    VAR Resultado = fact_model_match[win_target]
    RETURN
        IF(
            (DiffElo > 0 && Resultado = 1) ||
            (DiffElo < 0 && Resultado = 0),
            1.0,
            0.0
        )
)
```

Formate `Participação LAN`, `Taxa de Vitória` e `Vitória do Maior ELO` como porcentagem.

## Comportamento sob filtros

Uma medida é recalculada conforme os filtros. Se o usuário selecionar apenas 2025, `Partidas Únicas` passa a contar apenas 2025. Uma coluna calculada, por outro lado, é calculada linha a linha durante a atualização.

## Desempenho do modelo

As medidas abaixo já estão implementadas no projeto PBIP usando os nomes amigáveis do modelo semântico.

```DAX
Partidas no holdout = COUNTROWS(Previsoes)
```

```DAX
Acurácia do modelo = AVERAGE(Previsoes[Acerto])
```

```DAX
Brier médio = AVERAGE(Previsoes[Brier individual])
```

```DAX
Log-loss médio = AVERAGE(Previsoes[Log-loss individual])
```

```DAX
Erros de alta confiança = SUM(Previsoes[Erro de alta confianca])
```

O ROC-AUC do holdout aparece como medida global congelada (`0,6828385899814471`). Diferentemente da acurácia, o ROC-AUC não pode ser obtido corretamente apenas por uma média linha a linha; por isso, a medida não deve ser interpretada como recalculável para qualquer filtro.
