# Dados utilizados

Este repositório contém somente as entradas consolidadas necessárias à reprodução da camada analítica:

- `raw/core/matches_top50_dated.csv`: histórico operacional de partidas e equipes;
- `processed/match_feature_differences.csv`: recorte de modelagem com 38 atributos pré-jogo.

Os arquivos foram preparados no contexto acadêmico do TCC a partir de informações públicas da HLTV.org. Os scripts de coleta não estão incluídos. Os dados são disponibilizados para rastreabilidade do estudo; o uso posterior deve respeitar os termos e a origem das informações.

A licença MIT do repositório se aplica ao código autoral, não concede direitos sobre conteúdo de terceiros e não remove as condições aplicáveis à fonte original.

O campo `diff_maps_played_mean_5` representa experiência agregada dos jogadores e não identifica o mapa disputado. Portanto, a base não suporta análise por Mirage, Inferno, Nuke ou outros mapas específicos.
