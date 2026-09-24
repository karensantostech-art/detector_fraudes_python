# detector_fraudes_python

Modelos treinados para detectar fraudes a partir de um dataset desbalanceado.

# Detecção de Fraudes com Machine Learning

Projeto desenvolvido como parte de um bootcamp da DIO de IA Generativa, Dados e Cybersegurança, com o objetivo de aplicar conceitos de análise de dados, preparação de dados e Machine Learning utilizando Python para identificar transações financeiras fraudulentas.

## Objetivo

O objetivo deste projeto é desenvolver e comparar modelos de classificação capazes de identificar transações fraudulentas em um conjunto de dados de cartões de crédito.

Um dos principais desafios desse problema é o **desbalanceamento das classes**: a grande maioria das transações é legítima, enquanto uma pequena parcela corresponde a fraudes.

Nesse cenário, a acurácia, por si só, não é suficiente para avaliar o desempenho dos modelos. Um classificador poderia acertar quase todas as transações simplesmente prevendo que elas são legítimas, mas não conseguir identificar as fraudes, o que é um problema.

Por isso, foram utilizadas métricas como precisão (Precision), recall e F1-score da classe de fraude, além da análise das curvas ROC e Precision-Recall.

## Tecnologias e bibliotecas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- XGBoost
- SHAP
- Google Collab

## Conjunto de dados

Foi utilizado o conjunto de dados público de transações com cartões de crédito disponibilizado pelo TensorFlow.

[Dataset de transações financeiras](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)

O conjunto possui 31 colunas, incluindo variáveis anonimizadas, o valor da transação (`Amount`) e a variável-alvo (`Class`), que identifica transações legítimas (0) e fraudulentas (1).

A distribuição observada (proporção) foi de:

Transações legítimas (0) -> 99,83%
Transações fraudulentas (1) -> 0,17% 

## Etapas do projeto

### 1. Análise exploratória

Foi realizada uma análise inicial dos dados, incluindo a visualização das primeiras linhas e a investigação da distribuição da variável-alvo.

A análise identificou um forte desbalanceamento entre transações legítimas e fraudulentas, evidenciando a necessidade de avaliar os modelos com métricas específicas para a classe minoritária.

### 2. Preparação dos dados e engenharia de atributos

Foram realizadas as seguintes etapas:

- Criação de uma nova variável, `Amount_log`, utilizando a transformação logarítmica `log1p` sobre o valor da transação.
- Padronização da variável `Amount` utilizando `StandardScaler`, com a criação da coluna `Amount_scaled`.
- Remoção da coluna original `Amount` após a padronização.
- Separação das variáveis preditoras (`X`) e da variável-alvo (`y`).
- Divisão dos dados em conjuntos de treinamento e teste, utilizando 70% para treinamento e 30% para teste, com estratificação da variável-alvo.

### 3. Técnicas de balanceamento

Foram exploradas duas técnicas de balanceamento:

- **Undersampling:** redução da quantidade de amostras da classe majoritária.
- **SMOTE (Oversampling):** geração de amostras sintéticas da classe minoritária.

Essas técnicas foram estudadas durante o projeto, mas não foram utilizadas na comparação final dos modelos apresentada neste notebook.

### 4. Treinamento dos modelos

Foram implementados e avaliados quatro modelos:

1. Regressão Logística
2. Random Forest
3. Pipeline com padronização e Regressão Logística
4. XGBoost

Os modelos foram treinados com o conjunto de treinamento e avaliados no conjunto de teste.

### 5. Avaliação dos modelos

Foram utilizadas as seguintes métricas:

- **Precisão (Precision):** proporção das transações identificadas como fraudulentas que realmente eram fraudes.
- **Recall:** proporção das transações fraudulentas reais que foram identificadas pelo modelo.
- **F1-score:** média harmônica entre precisão e recall, permitindo avaliar o equilíbrio entre essas duas métricas.
- **ROC-AUC:** medida da capacidade de discriminação do modelo considerando diferentes limiares de classificação.

Também foram geradas curvas ROC e Precision-Recall para analisar o desempenho dos classificadores.

## Comparação dos modelos

Após o treinamento, foi realizada uma comparação com o limiar de decisão definido em 0,3, utilizando as probabilidades previstas por cada modelo.

Os resultados da análise com o limiar de 0.3 para a classe de fraude foram:

| Modelo | Precisão | Recall | F1-score |
--------------------------------------------------------
Regressão Logística | 78,57% | 66,89% | 72,26%
Random Forest | 63,16% | 81,08% | 71,01%
Pipeline | 79,51% | 65,54% | 71,85%
XGBoost | 90,70% | 79,05% | 84,48%
--------------------------------------------------------

Os resultados demonstraram diferenças no equilíbrio entre precisão e recall.

O Random Forest apresentou o maior recall entre os modelos avaliados com o limiar de 0,3, identificando uma proporção maior das fraudes existentes, porém com menor precisão.

O XGBoost apresentou o maior F1-score e a maior precisão nessa comparação, mantendo um recall de aproximadamente 79%.

## Limiar de decisão

Além da avaliação com o limiar padrão dos classificadores, foi realizada uma comparação utilizando o limiar de decisão de 0,3.

O limiar define a probabilidade mínima necessária para que uma transação seja classificada como fraude.

A redução do limiar em relação ao padrão de 0,5 permite que mais transações sejam sinalizadas como suspeitas, podendo aumentar a identificação de fraudes, mas também influenciando a quantidade de falsos positivos.

Para a comparação apresentada no projeto, foi utilizado o limiar de 0,3 em todos os modelos.

### Matriz de confusão do XGBoost

Com o limiar de 0,3, o XGBoost apresentou os seguintes resultados no conjunto de teste:

| | Predição: legítima | Predição: fraude |
|---|---:|---:|
| Real: legítima | 85.283 | 12 |
| Real: fraude | 31 | 117 |

O modelo identificou 117 das 148 transações fraudulentas presentes no conjunto de teste, enquanto 31 fraudes não foram identificadas.

Também foram classificadas incorretamente 12 transações legítimas como fraudulentas.

### Resultado do XGBoost

- Precisão: 90,70%
- Recall: 79,05%
- F1-score: 84,48%
- ROC-AUC: 96,87%

Esses resultados são referentes ao conjunto de teste e à configuração utilizada no notebook.

## Interpretabilidade com SHAP

Foi utilizada a biblioteca SHAP para investigar a importância das variáveis nas previsões do modelo XGBoost.

A análise foi realizada com uma amostra de 100 observações do conjunto de teste, utilizando `shap.Explainer` e um gráfico de barras para visualizar a importância global das variáveis.

O gráfico permite observar quais atributos tiveram maior contribuição absoluta nas previsões do modelo analisado.

Essa etapa foi incluída para complementar as métricas de desempenho com uma análise interpretativa do comportamento do classificador.

## Comparação com a abordagem da Expert

**Alterações realizadas em relação à implementação da Expert:**

> Em busca da convergência nos resultados das métricas dos modelos de Regressão Logística, o número máximo de iterações foi alterado de 1000 para 4500, já que 1000 iterações não era o suficiente para o treinamento completo dos modelos;

> Também completei as informações de cada modelo por criar os gráficos da curva ROC e da relação precisão-recall, para os 4 modelos criados, incluindo na versão final do modelo XGBoost ajustado com o threshold 0.3, para a análise completa de cada um deles;

> Enquanto a expert aplicou a mudança do valor threshold apenas no pipeline, decidi aplicar para os 4 modelos para a análise ficar mais completa; 

> Para melhor visualização de algumas estatísticas os resultados de determinadas métricas foram multiplicados por 100 e apresentados em porcentagem, essa alteração foi aplicada, por exemplo, à distribuição das classes, ao valor da ROC-AUC e às métricas de precisão, recall e F1-score utilizadas na comparação dos modelos.

## Resultados e conclusões

O projeto permitiu aplicar diferentes conceitos de Ciência de Dados e Machine Learning a um problema de classificação com classes fortemente desbalanceadas.

A comparação dos modelos demonstrou a importância de utilizar métricas além da acurácia, principalmente quando o objetivo é identificar corretamente a classe minoritária.

Entre as configurações avaliadas com o limiar de 0,3, o XGBoost apresentou o maior F1-score e a maior precisão, enquanto o Random Forest apresentou o maior recall.

A análise SHAP foi utilizada como ferramenta complementar para investigar a importância das variáveis nas previsões do modelo.

## Estrutura do repositório

```text
deteccao_fraudes_python/
│
├── README.md
└── detecção de fraudes DIO Bootcamp.ipynb
```

## Como executar

1. Clone este repositório.
2. Abra o notebook em um ambiente Jupyter, Google Colab ou outro ambiente compatível.
3. Instale as bibliotecas necessárias, caso ainda não estejam disponíveis.
4. Execute as células do notebook na ordem apresentada.

O conjunto de dados é carregado diretamente da URL pública utilizada no notebook, portanto é necessário ter acesso à internet para executar a etapa de carregamento.

## Observações

Este é um projeto educacional desenvolvido no contexto de um bootcamp da DIO de IA Generativa, Dados e Cybersegurança.

Os resultados apresentados são experimentais e correspondem à implementação e às configurações registradas no notebook. Eles não representam uma validação para uso em sistemas financeiros reais.
