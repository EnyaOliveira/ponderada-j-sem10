# Detecção de SPAM com LSTM + GloVe

Este projeto tem como objetivo implementar e comparar modelos de classificação de mensagens como **SPAM** ou **HAM**, utilizando redes neurais e diferentes técnicas de vetorização de texto.

## Descrição da Atividade

A atividade propõe:

1. Utilizar o dataset do Kaggle com mensagens rotuladas como `ham` ou `spam`;
2. Realizar análise exploratória e pré-processamento de linguagem natural (PLN);
3. Aplicar vetorização utilizando **GloVe** e treinar um modelo **LSTM**;
4. Avaliar o modelo com acurácia e matriz de confusão;
5. Treinar outro modelo de rede neural com **vetorização alternativa (TF-IDF)**;
6. Comparar os resultados entre os dois modelos.

---

## Modelos Desenvolvidos

### Modelo 1: LSTM + GloVe
- Vetorização densa com embeddings pré-treinados GloVe (`glove.6B.200d`)
- Arquitetura LSTM com camadas de Dropout
- Avaliação com acurácia e matriz de confusão

### Modelo 2: MLP + TF-IDF
- Vetorização TF-IDF (`max_features=5000`)
- Rede neural simples com duas camadas densas
- Avaliação com acurácia, matriz de confusão e relatório de classificação

---

## Resultados

| Modelo         | Vetorização | Acurácia | Observações                      |
|----------------|-------------|----------|----------------------------------|
| LSTM           | GloVe       | 97.9%    | Excelente desempenho geral       |
| MLP (Densa)    | TF-IDF      | ~94%     | Simples, rápido e eficaz         |

---

## Estrutura do Projeto
```
├── spam_detection_lstm_glove.ipynb 
├── README.md
├── glove.6B.200d.txt
└── spam.csv
``` 

## Como Executar

1. Baixe o dataset: [SMS Spam Collection - Kaggle]([https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download))
2. Baixe os embeddings: [GloVe 6B - 200d](https://nlp.stanford.edu/data/glove.6B.zip)
3. Execute o notebook `ponderada_J_sem10.ipynb` no Google Colab

---

## Requisitos

- Python 3.10+
- Bibliotecas: `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`, `scikit-learn`, `tensorflow`

---


