# calibracao-transformer-clinc150

Exemplo prático do seminário **"Calibração, incerteza e confiabilidade em modelos de PLN"** (disciplina de Deep Learning, PUC-Rio).

Um Transformer pequeno é treinado do zero para classificar as 150 intenções do CLINC150. Com ele, o exemplo mede três coisas:

1. **Calibração**: a confiança do softmax corresponde à taxa de acerto? (ECE antes e depois de temperature scaling)
2. **Erros de classificação**: a confiança separa acertos de erros? (predição seletiva e ensemble)
3. **Fora de escopo**: o modelo percebe frases que não pertencem a nenhuma das 150 intenções? (MSP, energia e distância de Mahalanobis)

Tudo roda em CPU, em cerca de 5 minutos por semente.

## Como rodar

```bash
pip install -r requirements.txt

# dados: data_full.json do repositório original do CLINC150
curl -L -o data_full.json https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json

# treina, calibra e mede a detecção de fora de escopo (uma semente por execução)
python3 calibracao.py data_full.json 0
python3 calibracao.py data_full.json 1
python3 calibracao.py data_full.json 2

# predição seletiva e ensemble, a partir dos logits salvos
python3 ensemble.py saida_semente0.npz saida_semente1.npz saida_semente2.npz
```

`calibracao.py` salva os logits em `saida_semente<N>.npz`. `ensemble.py` só lê esses arquivos, então roda em segundos.

## Arquivos

| Arquivo | O que faz |
| --- | --- |
| `calibracao.py` | Treina o modelo, ajusta a temperatura na validação, calcula o ECE e compara três pontuações de detecção de fora de escopo |
| `ensemble.py` | Calcula a predição seletiva, a AUROC de erro e o ensemble das sementes |
| `requirements.txt` | Dependências |

## O modelo

- **Dados**: CLINC150, com 15.000 frases de treino, 3.000 de validação, 4.500 de teste e 1.000 frases de teste fora do escopo. As frases fora do escopo não entram no treino.
- **Entrada**: palavras inteiras em minúsculas, até 30 por frase. O vocabulário vem do treino (cerca de 5.200 palavras).
- **Arquitetura**: codificador Transformer com 2 camadas, 4 cabeças de atenção, vetores de 128 dimensões e camada interna de 256. A frase é representada pela média dos vetores das palavras e passa por uma camada linear com 150 saídas.
- **Tamanho**: 956.694 parâmetros.
- **Treino**: do zero, sem pré-treino. Entropia cruzada, AdamW, 20 épocas, lotes de 64.

## Resultados

Média e desvio padrão de três sementes (0, 1 e 2).

### Calibração

| | Sem calibração | Com temperature scaling |
| --- | --- | --- |
| Acurácia | 89,1 ± 0,2% | 89,1 ± 0,2% |
| Confiança média | 94,5 ± 0,1% | 88,7 ± 0,2% |
| ECE (10 faixas) | 5,50 ± 0,15% | 1,22 ± 0,05% |

A temperatura ajustada foi T = 1,62 ± 0,01. O modelo é sobreconfiante, e dividir os logits por T corrige a confiança sem mudar a acurácia.

### A confiança separa acerto de erro?

| | Uma semente (média) | Ensemble das 3 |
| --- | --- | --- |
| Acurácia respondendo tudo | 89,1% | 91,3% |
| Acurácia nas 90% mais confiantes | 95,1% | 96,5% |
| Acurácia nas 80% mais confiantes | 97,8% | 98,3% |
| AUROC de erro | 93,1 | 93,3 |

Temperature scaling quase não altera esses números (AUROC de erro de 93,1 para 93,0): ele muda o valor da confiança, não a ordem das previsões.

As três sementes discordam em 13,8% das frases de teste. O ensemble acerta 97,2% quando elas concordam e 54,3% quando discordam.

### Detecção de fora de escopo

| Pontuação | AUROC | FPR@95 |
| --- | --- | --- |
| MSP (confiança do softmax) | 91,1 | 29,3 |
| Energia | 93,6 | 22,4 |
| Mahalanobis | 92,2 | 28,5 |

FPR@95 é a fração de frases dentro do escopo barradas quando se detectam 95% das frases de fora. Cerca de 20% das frases fora do escopo recebem mais de 90% de confiança.

### Exemplos (semente 0)

| Frase | Previsão | Rótulo | Confiança |
| --- | --- | --- | --- |
| how do you say fast in spanish | translate | translate | > 99,9% |
| what is some trivia about the weather | weather | fun_fact | 99,95% |
| who invented the internet | who_made_you | fora do escopo | 99,96% |
| what is my current location | current_location | share_location | 99,99% |

## Limitações

- O modelo é pequeno e não tem pré-treino, então é mais sobreconfiante do que um BERT ajustado. O exemplo ilustra o fenômeno; não mede o estado da arte.
- Os exemplos da tabela acima foram escolhidos por serem ilustrativos.
- Label smoothing não foi testado.

## Referências

- Larson et al. (2019). *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*. EMNLP-IJCNLP. Dados: https://github.com/clinc/oos-eval
- Guo et al. (2017). *On Calibration of Modern Neural Networks*. ICML.
- Desai e Durrett (2020). *Calibration of Pre-trained Transformers*. EMNLP.
- Podolskiy et al. (2021). *Revisiting Mahalanobis Distance for Transformer-Based Out-of-Domain Detection*. AAAI.

## Licença

O código está sob a licença GPL-3.0 (arquivo `LICENSE`). O CLINC150 não faz parte deste repositório e é distribuído pelos autores sob a licença Creative Commons Attribution 3.0.
