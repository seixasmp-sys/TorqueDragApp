# T&D Complete App

Aplicacao Streamlit integrada para comparacao de modelos, analise de sensibilidade e ajuste de modelos de Torque & Drag com dados experimentais.

## Como executar no Windows

1. Extraia a pasta do ZIP.
2. Clique duas vezes em `run_app.bat`.

O arquivo `.bat` instala as dependencias necessarias automaticamente.

## Bibliotecas

A aplicacao usa arquivos JSON locais na mesma pasta do `app.py`:

- `td_models_library.json`: modelos simbolicos de mueff e Fd/(Wbp L)
- `td_campaign_library.json`: campanhas experimentais
- `td_experiment_library.json`: experimentos vinculados a uma campanha
- `td_analysis_library.json`: resultados de ajuste por least_squares

## Novos modulos experimentais

### Campaign Library

Cria campanhas com:

- nome da campanha
- rho_f em ppg
- rho_p em kg/m3
- w_pp em N/m
- OD em in
- ID em in
- L em ft

### Experiment Library

Cria experimentos vinculados obrigatoriamente a uma unica campanha. O modulo calcula automaticamente:

- A = Ac/(Aw - Ac)
- w_bp = w_pp * (1 - rho_f/rho_p)

A correcao do primeiro caso do calculo de A foi implementada usando R2 na expressao do segmento circular externo.

Os dados experimentais devem ser informados como:

- x em ft
- FD em N

A aplicacao converte internamente:

- x_i = x/L
- FD/(w_bp L), usando L convertido de ft para m, pois w_bp esta em N/m

### Model Fitting Analysis

Ajusta mu1 e mu2 para um ou mais modelos usando `scipy.optimize.least_squares`.

Configuracao padrao do ajuste:

- chute inicial: mu1 = 0.25, mu2 = 0.5
- limites: 0 < mu1 < 1 e 0 < mu2 < 1
- metricas salvas: mu1, mu2, R2, RMSE

### Option 1 - Model Fit by Campaign

Seleciona uma campanha e um modelo. Mostra uma tabela com todos os experimentos da campanha que possuem ajuste salvo para aquele modelo.

Tambem permite visualizar e exportar em PNG o grafico com dados experimentais e modelo ajustado para um experimento escolhido.

Exports:

- tabela em `.xlsx`
- tabela em `.txt`
- grafico em `.png`

### Option 2 - Compare Fitted Models

Seleciona campanha, experimento e modelos. Mostra os dados experimentais e as curvas dos modelos ajustados.

Exports:

- grafico em `.png`
- tabela em `.xlsx`
- tabela em `.txt`

## Biblioteca de modelos

Cada modelo possui a propriedade `force_direction`:

- `opposite`: usa a convencao com integral negativa

  Fd/(Wbp L) = - integral(mu_eff dxi)

- `same`: usa integral positiva

  Fd/(Wbp L) = + integral(mu_eff dxi)

Essa opcao pode ser escolhida na tela Shared Model Library ao criar ou editar um modelo.
