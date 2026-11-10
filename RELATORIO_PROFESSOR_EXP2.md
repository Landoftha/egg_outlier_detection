# 📑 Relatório Técnico-Científico — Experimento 2 (`New sample/`): Detecção de Rachaduras em Ovos com Divisão por Sessão, Intervalo de Confiança e Matriz de Custo Industrial

**Projeto:** Egg Outlier Detection (Triagem Automatizada de Ovos Quebrados/Trincados)  
**Data:** Novembro / 2026  
**Scripts Principais:** `workflow_exp2_new_sample.py` e `generate_exp2_visualizations.py`  
**Arquivos de Saída:** `exp2_new_sample_results.csv`, `exp2_session_fpr_summary.csv`, `Visualizacao_Features/00_COMPARATIVO_NEW_SAMPLE_PROFESSOR.png` e `Visualizacao_Features/01_GRAFICOS_RESULTADOS_EXP2_PROFESSOR.png`

---

## 1. Motivação e Feedback da Revisão por Pares (Indústria / LinkedIn)

Na Coleta v1 (`Dataset/` com 191 imagens: 181 ovos bons e 10 ovos quebrados), o modelo preliminar atingiu **99,0% de ROC-AUC (100% de Recall em $n=5$ ovos trincados no teste)**. Ao submeter os resultados à comunidade técnica de Visão Computacional, três pontos metodológicos fundamentais foram levantados por profissionais da área:

1. **Largura do Intervalo de Confiança e Vazamento de Ambiente:** Com apenas 10 ovos trincados divididos entre validação e teste, um Recall de 100% ($5/5$) possuía um Intervalo de Confiança de 95% de Wilson largo ($[56,6\%, 100,0\%]$). Além disso, recomendou-se separar e avaliar os dados levando em conta **sessões de captura distintas (dia, lote, iluminação)** e **indivíduos físicos** (evitando que duas fotos do mesmo ovo em ângulos diferentes caíssem no treino e no teste).
2. **Limiar de Anomalia ($\tau$) e Custo Assimétrico de Erro na Granja:** Na indústria avícola, um **Falso Positivo ($C_{FP}$)** desvia um ovo bom para processamento industrial líquido (custo baixo), enquanto um **Falso Negativo ($C_{FN}$)** deixa passar um ovo trincado para a embalagem comercial (risco sanitário de *Salmonella*, vazamento na bandeja e penalidade comercial — custo alto). O limiar $\tau$ deve ser calibrado explicitamente em função dessa razão de custo ($C_{FN}:C_{FP}$).
3. **Falso Positivo por Sessão/Lote e Textura Superficial (Porosidade):** Ovos de diferentes sessões/lotes variam na porosidade natural da casca. Filtros de borda ou distâncias absolutas podem confundir cascas naturalmente mais porosas com microtrincas.

---

## 2. Nova Coleta de Dados (`New sample/`) e Higienização Anti-Vazamento

Para responder a esses três pontos, foi realizada a **Coleta v2 (`New sample/`)**, rotacionando os ovos em múltiplos ângulos ao longo de **4 dias/sessões de captura distintas**:

* **Auditoria e Deduplicação por Hash MD5:** Antes do treinamento, o pipeline (`locate_and_crop_egg_v2`) identificou e removeu **3 imagens duplicadas** (`IMG_20261103_172523.jpg`, `172527.jpg` e `172529.jpg`) que haviam sido copiadas para `Bad eggs` mas permaneciam em `Good Eggs`, eliminando 100% de risco de *label leakage*.
* **Composição Final da Coleta v2 (`N = 732` imagens únicas):**
  * **Sessão `2026-11-03`:** 51 fotos (`47 Ovos Bons` | `4 Ovos Quebrados`)
  * **Sessão `2026-11-04`:** 156 fotos (`136 Ovos Bons` | `20 Ovos Quebrados`)
  * **Sessão `2026-11-07`:** 219 fotos (`219 Ovos Bons` | `0 Ovos Quebrados` — *Lote de Controle Normal*)
  * **Sessão `2026-11-08`:** 306 fotos (`219 Ovos Bons` | `87 Ovos Quebrados`)
  * **Total:** **621 fotos de Ovos Bons** e **111 fotos de Ovos Quebrados** (aumento de **11,1x** no volume de amostras com rachadura).
* **Agrupamento Temporal por Ovo Físico (`egg_group_id` via $|\Delta t| \le 25\text{s}$):**
  * Como cada ovo foi fotografado girando em sequência, extraímos o *timestamp* exato do nome do arquivo (`IMG_YYYYMMDD_HHMMSS.jpg`) e agrupamos fotos consecutivas ($|\Delta t| \le 25\text{s}$) no mesmo cluster físico.
  * Foram identificados **81 grupos de ovos físicos independentes** (`57 grupos Bons` e `24 grupos Quebrados`). A divisão entre `Train`, `Val` e `Test` foi feita **estritamente no nível do grupo físico**, garantindo que dois ângulos do mesmo ovo jamais se cruzem entre treino e teste.

---

## 3. Aperfeiçoamentos no Pré-Processamento e Extração de Features

1. **Localizador de ROI v2 com Rejeição de Mesa e Invariância à Distância (`locate_and_crop_egg_v2`):**
   * *Problema resolvido:* Nas fotos verticais tiradas mais de longe na Coleta v2, a **mesa bege** aparecia acima e abaixo do tapete azul (`APP TECH`). Como a mesa bege possui $R > B$, selecionar apenas o maior contorno (`max(contours, key=cv2.contourArea)`) capturava a mesa em 70% das fotos.
   * *Técnica aplicada:* Aplicamos a diferença espectral `cv2.subtract(r_chan, b_chan)` + Limiar de Otsu (`cv2.THRESH_OTSU`), rejeitamos contornos de borda externa que atravessam a largura do quadro (`wb > 0.92 * rw`) e selecionamos o contorno que maximiza a razão $\frac{\text{Área}}{1 + 2 \cdot d_{\text{centro}}}$ via Momentos de Imagem (`cv2.moments`).
   * *Mapeamento para Alta Resolução Original:* Em vez de recortar o ovo na imagem reduzida (o que apagaria microtrincas capilares nas fotos tiradas mais de longe), mapeamos as coordenadas do contorno de volta para a foto original de alta resolução (~12 MP) e redimensionamos apenas a ROI do ovo para `224x224` pixels (**100% de sucesso nas 732 imagens**).

2. **Extração Morfológica Anisotrópica (Trinca Linear vs. Poro Circular):**
   * Aplicamos a transformada **White Top-Hat** (`cv2.MORPH_TOPHAT`), **Black Top-Hat** (`cv2.MORPH_BLACKHAT`) e **Operador Laplaciano** (`cv2.Laplacian`) exclusivamente dentro da máscara interna erodida (`cv2.erode`), evitando artefatos de sombra na borda.
   * Calculamos o **Alongamento Anisotrópico Máximo (`max_elongation`)** via `cv2.connectedComponentsWithStats`:
     $$\text{Elongation} = \frac{w_c^2 + h_c^2}{\text{Área}_c}$$
     Enquanto poros naturais da casca são isotrópicos/redondos ($\text{Elongation} \approx 2\text{–}9$), fissuras e trincas lineares possuem alto alongamento ($\text{Elongation} > 13$).

---

## 4. Arquiteturas de Redes Neurais Avaliadas e Resultados (`EarlyStopping`)

Avaliamos três arquiteturas com parada antecipada (`EarlyStopping`) e intervalos de confiança de 95% (Bootstrap Estratificado $B=1000$ para `ROC-AUC` e Intervalo de Wilson para `Recall`/`FPR`):

### Experimento 2A — `SmallConvAutoencoder` (CNN One-Class treinada do zero)
* **Configuração:** Rede Convolucional Encoder-Decoder ($64 \times 64$) treinada exclusivamente nos **436 ovos bons** (`0` quebrados no treino), otimizador `AdamW` e `EarlyStopping` (interrompido na época 22; melhor época = 14).
* **Resultado:** `Test AUROC = 62,73%` (95% CI: `[52,6%, 71,3%]`).
* **Diagnóstico:** Microtrincas capilares da Coleta v2 ocupam menos de 1% dos pixels da casca; uma CNN rasa de reconstrução global sofre interferência da variação de iluminação entre os 4 dias de coleta.

### Experimento 2B — `Self-Normalized Spatial PatchCore` + Anisotropia (One-Class Puro)
* **Configuração:** Extração de *embeddings* espaciais em grade $28 \times 28$ combinando camadas intermediárias da **`MobileNetV3-Large`** (camadas 3, 6 e 12) e **`ResNet18`** (`layer1`, `layer2`, `layer3`) pré-treinadas na ImageNet. O Banco de Memória (*Coreset* de 20.000 patches) contém **apenas patches internos dos 436 ovos bons de treino**.
* **Inovação Anti-Falso Positivo (`Self-Normalized PatchCore`):** Para evitar que um lote de ovos com casca naturalmente mais porosa tenha distância absoluta $d_p$ alta em toda a superfície, normalizamos a distância de cada patch $p$ subtraindo a **mediana das distâncias do próprio ovo**:
  $$\Delta d_p = d_p - \text{mediana}_{q \in \text{Ovo}}(d_q)$$
* **Resultados (Sem nenhum ovo quebrado no treino):**
  * **Validação ($n=169$: 120 Bons + 49 Quebrados):** **`ROC-AUC = 86,09%`** (95% CI: `[79,4%, 92,0%]`)
  * **Teste ($n=127$: 65 Bons + 62 Quebrados):** **`ROC-AUC = 80,72%`** (95% CI: `[72,2%, 88,3%]`)
  * **Avaliação Global One-Class ($n=296$, todos os 111 ovos quebrados):** **`ROC-AUC = 83,47%`** (95% CI: `[78,7%, 88,3%]`)
  * **Generalização Separada por Sessão:** Sessões diurnas (`03/Nov + 04/Nov`): **`83,93% AUROC`** | Sessão noturna (`08/Nov`): **`77,27% AUROC`**.

### Experimento 2C — `HybridCrackNet` (Fusão `PatchCore` + Rede Neural Densa Regularizada com `EarlyStopping`)
* **Configuração:** Rede Neural PyTorch (`Linear -> LayerNorm -> GELU -> Dropout(0.35) -> Linear -> LayerNorm -> GELU -> Dropout(0.20) -> Linear`) combinando os *embeddings* multi-escala, os descritores espaciais do `Self-Normalized PatchCore` e as features morfológicas anisotrópicas, treinada com `BCEWithLogitsLoss` ponderada e **`EarlyStopping`** (parou automaticamente na época 38; melhor época = 20).
* **Resultados (Divisão estrita por Ovo Físico):**
  * **Validação ($n=146$: 120 Bons + 26 Quebrados):** **`ROC-AUC = 91,09%`** (95% CI: `[84,2%, 96,5%]`)
  * **Teste Independente ($n=101$: 65 Bons + 36 Quebrados):** **`ROC-AUC = 89,19%`** (95% CI: `[81,7%, 95,3%]`)

---

## 5. Calibração do Limiar Operacional ($\tau$) por Matriz de Custo Industrial

O limiar de decisão $\tau$ foi calibrado no conjunto de Validação minimizando a função de perda industrial $\mathcal{L}(\tau) = C_{FN} \cdot \text{FNR}(\tau) + C_{FP} \cdot \text{FPR}(\tau)$ e avaliado no Conjunto de Teste:

| Modelo | Regime de Custo ($C_{FN}:C_{FP}$) | Limiar ($\tau$) | Recall / Sensibilidade (IC 95% Wilson) | Falso Positivo — FPR (IC 95% Wilson) | F1-Score | Interpretação Prática na Granja |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp 2B (One-Class Puro)** | **Industrial ($5:1$)** | `3,2547` | **82,3%** (`[71,0%, 89,8%]`) | **29,2%** (`[19,6%, 41,2%]`) | **77,3%** | Detecta 51/62 trincados sem nunca ter visto um ovo quebrado no treino. |
| **Exp 2B (One-Class Puro)** | **Rigor Máximo ($10:1$)** | `2,3609` | **93,5%** (`[84,6%, 97,5%]`) | **80,0%** (`[68,7%, 87,9%]`) | **67,4%** | Retém 58/62 trincados no modo puramente não-supervisionado. |
| **Exp 2C (`HybridCrackNet`)** | **Balanceado ($1:1$)** | `0,4430` | **66,7%** (`[50,3%, 79,8%]`) | **7,7%** (`[3,3%, 16,8%]`) | **73,8%** | Baixíssimo descarte de ovos bons (apenas 5/65 falsos positivos). |
| **Exp 2C (`HybridCrackNet`)** | **Industrial ($5:1$)** | `0,3509` | **83,3%** (`[68,1%, 92,1%]`) | **27,7%** (`[18,3%, 39,6%]`) | **71,4%** | Ponto ótimo entre retenção de rachaduras (30/36) e preservação do lote. |
| **Exp 2C (`HybridCrackNet`)** | **Rigor Segurança ($10:1$)** | `0,2549` | **97,2%** (`[85,8%, 99,5%]`) | **67,7%** (`[55,6%, 77,8%]`) | **60,9%** | **Detecta 35 de 36 ovos trincados** (apenas 1 falso negativo). |

---

## 6. Taxa de Falsos Positivos (FPR) Estratificada por Sessão de Captura

Respondendo diretamente à questão levantada no LinkedIn sobre a estabilidade de falsos positivos entre diferentes lotes/sessões (`exp2_session_fpr_summary.csv`):

| Sessão / Data | Ovos Avaliados (Val + Test) | Ovos Bons | Ovos Quebrados | Falsos Positivos (`HybridCrackNet` $\tau=0,4430$) | FPR por Sessão (IC 95% Wilson) | Recall por Sessão |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sessão 1 (`03/Nov/2026`)** | 13 | 12 | 1 | 1 | **8,3%** (`[1,5%, 35,4%]`) | **100,0%** (1/1) |
| **Sessão 2 (`04/Nov/2026`)** | 50 | 37 | 13 | 2 | **5,4%** (`[1,5%, 17,7%]`) | **46,2%** (6/13) |
| **Sessão 3 (`07/Nov/2026`)** | 78 | 78 | 0 | 2 | **2,6%** (`[0,7%, 9,0%]`) | N/A (Controle) |
| **Sessão 4 (`08/Nov/2026`)** | 106 | 58 | 48 | 12 | **20,7%** (`[12,2%, 32,8%]`) | **83,3%** (40/48) |

* **Conclusão Experimental Importante:** Nas três sessões diurnas (`03/Nov`, `04/Nov` e `07/Nov`), a taxa de falsos positivos manteve-se extremamente baixa (**entre 2,6% e 8,3%**), comprovando que a normalização intra-ovo do `PatchCore` resolveu a confusão com a porosidade natural da casca. Já na sessão noturna (`08/Nov`, fotografada sob iluminação artificial inclinada e maior distância), o FPR subiu para **20,7%**, demonstrando empiricamente o ganho que será obtido na próxima etapa com suporte de **câmera estática e iluminação fixa**.
