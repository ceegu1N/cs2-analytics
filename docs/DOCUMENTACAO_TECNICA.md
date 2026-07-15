# Documentação técnica

O CS2 Analytics é uma extensão analítica do TCC sobre predição pré-jogo de partidas profissionais de Counter-Strike 2. O projeto transforma bases consolidadas em um banco SQLite, responde perguntas com SQL e apresenta os resultados em dashboards HTML/SVG e Power BI.

## Arquitetura

```text
CSVs consolidados + modelo congelado
                |
                v
       validação e preparação em Python
                |
                v
      SQLite: dimensões e tabelas fato
         |            |             |
         v            v             v
  consultas SQL   Power BI     HTML/SVG e PNG
         |
         v
  CSVs de resultado e conclusões
```

As responsabilidades são separadas:

- **Python:** valida entradas, constrói o banco, reaplica o modelo e automatiza as saídas;
- **SQLite:** armazena dimensões, partidas e previsões em um arquivo local;
- **SQL:** filtra, combina e resume os dados por meio de nove consultas versionadas;
- **Power BI e DAX:** organizam o modelo semântico e calculam indicadores sob filtros;
- **HTML/SVG:** oferece uma versão web interativa sem dependências externas de visualização.

`run_all.py` coordena o fluxo. A implementação fica dividida nos módulos de `src/cs2_sql_analytics`, enquanto `config.py` centraliza os caminhos relativos ao repositório.

## Modelo de dados

| Tabela | Granularidade | Uso principal |
|---|---|---|
| `dim_team` | uma equipe normalizada | nomes e filtros por equipe |
| `dim_date` | uma data | análises por mês, trimestre e semestre |
| `fact_match` | uma partida única | volume, ambiente e formato |
| `fact_team_match` | uma equipe em uma partida | vitórias e desempenho por equipe |
| `fact_model_match` | uma partida do recorte experimental | treino, holdout e 38 atributos pré-jogo |
| `fact_prediction_holdout` | uma previsão do modelo final | métricas e erros no holdout |

A separação entre `fact_match` e `fact_team_match` evita contar duas perspectivas de equipe como duas partidas. Chaves estrangeiras relacionam fatos às dimensões, e os testes verificam integridade, unicidade e granularidade.

## Execução

Instale as dependências e reconstrua o projeto:

```powershell
python -m pip install -r requirements.txt
python run_all.py --rebuild
```

O comando:

1. cria `output/cs2_analytics.sqlite`;
2. executa as nove consultas de `sql/queries`;
3. exporta resultados e seis tabelas para o Power BI;
4. atualiza o parâmetro portátil `DataRoot`;
5. gera cinco PNGs, o dashboard HTML e conclusões não técnicas.

Uma consulta isolada pode ser executada com `python run_query.py 02_desempenho_equipes`. A suíte completa é executada com:

```powershell
python -m unittest discover -s tests -v
```

## SQL e resultados

`sql/schema.sql` define o modelo relacional. Os arquivos de `sql/queries` foram escritos manualmente; Python apenas os envia ao SQLite e exporta as respostas. As consultas cobrem visão geral, equipes, evolução mensal, LAN/online, formatos, confrontos diretos, faixas de ELO, treino/holdout e ranking com função de janela.

Os CSVs de `output/query_results` usam ponto e vírgula, vírgula decimal e UTF-8 para facilitar a abertura em ambientes configurados em português.

## Dashboards

O dashboard web está em `output/dashboard/index.html`. Seus gráficos são SVGs gerados em Python e incluem tooltips, tema claro/escuro e tabelas auxiliares. Os cinco PNGs servem como prévia estática e como alternativa para documentação.

O relatório Power BI é aberto por `power_bi/CS2_Analytics.pbip`. O diretório `CS2_Analytics.SemanticModel` contém tabelas, relacionamentos e medidas; `CS2_Analytics.Report` contém as cinco páginas e seus visuais. As fórmulas principais estão resumidas em [`medidas_dax.md`](../power_bi/medidas_dax.md).

Após executar o pipeline, abra o projeto no Power BI Desktop e selecione **Atualizar**. Caso necessário, confira `DataRoot` em **Transformar dados > Gerenciar parâmetros**.

## Previsões e validação

`build_predictions.py` carrega `models/logistic_regression.joblib` e reaplica o modelo final às 294 partidas do holdout. Essa etapa não retreina nem altera a regressão logística: ela reproduz probabilidades, classes, matriz de confusão, Brier, log-loss e erros de alta confiança.

Os 20 testes automatizados verificam construção do banco, chaves, granularidade, execução das consultas, métricas congeladas, exportações para Power BI, cinco páginas do dashboard, equipes exibidas e portabilidade dos caminhos.

## Limites de interpretação

- a base operacional acompanha o recorte de equipes e fontes do projeto original;
- ELO e demais indicadores representam informações disponíveis antes das partidas;
- análises por contexto mostram associações, não relações causais;
- o dashboard complementa o TCC, mas não substitui seu protocolo experimental;
- probabilidades indicam favoritismo estimado, não garantia de resultado.
