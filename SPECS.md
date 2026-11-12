# 📋 Especificações do Projeto e Diretrizes para Agentes de IA (`SPECS.md`)

> **Propósito deste documento:** Este arquivo é a fonte oficial de verdade (*Single Source of Truth*) do projeto **Egg Outlier Detection**. Qualquer pessoa ou Agente de Inteligência Artificial que clonar este repositório **DEVE** ler e seguir rigorosamente estas especificações antes de sugerir, criar ou alterar qualquer código.

---

## 🎯 1. Objetivo Central e Contexto Industrial

* **Problema:** Detecção automatizada de anomalias em ovos, **especificamente ovos quebrados ou com rachaduras (trincados)**, utilizando treinamento de **Redes Neurais**.
* **Aplicação Prática:** Facilitar e automatizar a triagem e separação de ovos na indústria avícola. Para que os ovos possam ser comercializados, **todos devem estar em perfeitas condições estruturais**.
* **Natureza do Trabalho:** Este projeto é conduzido como um **experimento científico iterativo**. O objetivo é testar diferentes arquiteturas de modelos (iniciando com modelos pequenos para validar resultados e evoluindo para um dataset mais robusto), criando *workflows* específicos para cada modelo e documentando o desempenho de cada abordagem.

---

## 🚧 2. Escopo Estrito (O que fazer vs. O que NÃO fazer)

### ✅ O que está DENTRO do escopo atual:
1. **Identificação/Localização do ovo** na imagem (isolar o ovo do fundo).
2. **Detecção exclusiva de RACHADURAS e QUEBRAS** na casca do ovo via Redes Neurais.
3. **Treinamento no paradigma de Detecção de Anomalias (*One-Class Learning*):**
   * **Treino (`train`):** Composto **exclusivamente por ovos bons** (sem rachaduras).
   * **Validação (`val`) e Teste (`test`):** Conterão os **ovos ruins (com rachaduras)** — rotulados manualmente pela autora — junto com amostras de ovos bons para medir a capacidade real de separação.
4. **Uso de Parada Antecipada (*Early Stopping*):** Todo treinamento de rede neural deve obrigatoriamente conter *Early Stopping* configurado para interromper a execução caso a métrica de validação pare de melhorar ao longo do tempo (épocas), evitando gasto computacional desnecessário e *overfitting*.

### ❌ O que está FORA do escopo atual (Deixado para etapa futura):
* **Pintinhas naturais da casca (sardas) e sujeira superficial:**
  * **Motivo técnico:** Nos testes anteriores, as pintinhas e sujeiras se confundiram com as rachaduras durante o treinamento. Portanto, **pintinhas e sujeira foram removidas do escopo deste modelo** e serão tratadas futuramente em um modelo separado e exclusivo para limpeza/pigmentação.

---

## 🛠️ 3. Bibliotecas e Tecnologias Padrão

As implementações devem priorizar o ecossistema abaixo:

* **`anomalib`**: Biblioteca principal planejada para o treinamento, avaliação e *workflows* de modelos de detecção de anomalias em imagens (ex: *Padim*, *PatchCore*, *FastFlow*, *Autoencoders*).
* **`PyTorch` (`torch`, `torchvision`)** & **`Lightning` (`pytorch-lightning`)**: Backend de redes neurais, *callbacks* de *Early Stopping* (`EarlyStopping`) e carregamento de dados (`DataLoader`).
* **`OpenCV` (`cv2`)**: Localização/recorte do ovo (*ROI*), pré-processamento e manipulação de máscaras.
* **`Scikit-Learn` (`sklearn`)**: Métricas de avaliação (`roc_auc_score`, `f1_score`, `precision_recall_curve`, `confusion_matrix`) e modelos de *baseline*.
* **`NumPy`, `Pandas` & `Matplotlib`**: Estruturação de tabelas de experimentos e visualização de mapas de calor de anomalias (*anomaly heatmaps*).

---

## 🔬 4. Metodologia de Treinamento e Dados

1. **Divisão dos Dados (Regra de Ouro):**
   * `dataset/train/good/` $\rightarrow$ Apenas ovos em perfeitas condições. A rede neural aprende a distribuição de normalidade.
   * `dataset/val/good/` e `dataset/val/cracked/` $\rightarrow$ Usados a cada época para calibrar o limiar de anomalia e acionar o **`EarlyStopping`**.
   * `dataset/test/good/` e `dataset/test/cracked/` $\rightarrow$ Usados apenas ao final para gerar as métricas definitivas do experimento.
2. **Rotulagem Manual:**
   * A autora do projeto fornecerá a identificação/rotulagem exata de quais ovos possuem rachadura para compor as pastas de validação e teste.
3. **Estratégia Inicial:**
   * Começar treinando um **modelo pequeno e leve** nos dados disponíveis para validar o pipeline de ponta a ponta e observar os resultados antes da coleta do dataset definitivo.

---

## 📓 5. Registro de Achados e Lições Aprendidas (O que funcionou vs. O que falhou)

Este histórico deve ser consultado para não repetir abordagens que já se provaram ineficazes e deve ser atualizado a cada novo experimento:

### ❌ O que NÃO funcionou (Erros e Limitações Identificadas):
1. **Fotos tiradas de forma não-estática (Coleta v1):**
   * *Problema:* As primeiras 200 fotos foram tiradas manualmente sem suporte fixo, gerando variações de distância da câmera, ângulo de inclinação e iluminação (sombras e reflexos da lâmpada em posições diferentes).
   * *Impacto:* Algoritmos baseados em área, contorno e gradiente luminoso sofreram interferência da mudança de posição da câmera.
2. **Ausência de fotos girando o ovo (Falso positivo por orientação):**
   * *Problema:* Quando o ovo era fotografado de lado ou levemente inclinado, os extratores geométricos achavam que o ovo era deformado/ruim.
   * *Solução aplicada na Coleta v2 (`New sample/`):* Foram coletadas **732 imagens únicas** (621 fotos de ovos bons e 111 fotos de ovos quebrados) ao longo de **4 dias/sessões de captura** (`03/Nov`, `04/Nov`, `07/Nov` e `08/Nov/2026`), girando os ovos em múltiplos ângulos, com **câmera fixa na mesma posição e iluminação fixa** em todas as sessões. Entre sessões variou apenas o **zoom** (as fotos de `04/Nov` estão mais aproximadas).
   * **⚠️ Trincas desenhadas a lápis (sessão `08/Nov`):** as **87 fotos de ovos "quebrados" de `08/Nov` (17 ovos físicos) são trincas DESENHADAS A LÁPIS**, criadas pela autora para contornar a falta de ovos realmente trincados. Trincas **reais** existem apenas em `03/Nov` e `04/Nov` (**24 fotos, 7 ovos físicos**). Regra a partir do Exp 3: trinca a lápis só pode entrar no **treino** (e num relatório à parte); **validação e teste usam apenas trincas reais**.
3. **Misturar pintinhas/sujeira com rachaduras no mesmo detector:**
   * *Problema:* Filtros de borda (como Canny) e desvios de cor ativavam fortemente nas pintinhas naturais da casca marrom, gerando falsos positivos de trinca.
   * *Solução definida:* Isolar o problema — focar agora 100% em rachaduras/quebras e deixar pintinhas/sujeira para outro modelo futuro.
4. **Limiarização simples (Otsu em escala de cinza) ou maior contorno sem filtro de borda quando a mesa aparece no enquadramento:**
   * *Problema:* Na Coleta v2 (`New sample/`), como algumas fotos foram tiradas mais de longe mostrando a mesa bege ao redor do tapete azul, escolher apenas `max(contours, key=cv2.contourArea)` no canal $R - B$ selecionava a faixa da mesa bege em 70% das fotos.
   * *Solução implementada (`locate_and_crop_egg_v2`):* Rejeição de contornos de borda externa (`wb > 0.92 * rw`), ponderação por centralidade espacial e mapeamento do contorno de volta para a resolução original antes do recorte (`224x224`), atingindo **100% de sucesso (732/732 fotos)**.
5. **Falsos positivos por porosidade/textura superficial entre sessões/lotes diferentes:**
   * *Problema apontado nos comentários do LinkedIn:* Ovos de diferentes sessões/lotes apresentam variações naturais na textura e porosidade da casca. O cálculo de distância absoluta no *PatchCore* ou filtros globais eleva a taxa de falsos positivos (FPR) em cascas mais porosas.
   * *Solução implementada:* **Self-Normalized Spatial PatchCore** (subtração da mediana intra-ovo $\Delta d_p = d_p - \text{mediana}_{q \in \text{Ovo}}(d_q)$ sobre grade $28 \times 28$ com máscara erodida) combinado com **Anisotropia de Componentes Conectados** (`max_elongation`), que diferencia microtrincas lineares alongadas de poros circulares.

6. **Avaliar com trincas desenhadas a lápis (Exp 2):**
   * *Problema:* Validação e teste do Exp 2 misturavam trincas reais (03–04/Nov) com trincas a lápis (08/Nov), inflando as métricas. Nas trincas reais de 04/Nov, o recall do Exp 2C foi de só 46,2% (6/13).
   * *Solução (Exp 3):* lápis só no treino; validação e teste só com trincas reais.
7. **Treinar o classificador com trincas a lápis (Exp 3, modo Híbrido):**
   * *Problema:* A regressão logística treinada com trincas a lápis ficou **pior** nas trincas reais que o modelo One-Class (teste: mediana 79,5% vs. 89,3% em 25 divisões). O lápis não representa bem uma trinca real.
8. **Vazamento nas features de treino do Exp 2C:** os ovos de treino eram comparados com um banco de memória que os continha. *Solução (Exp 3):* features *out-of-fold*.
9. **Limiar por custo sem prevalência (Exp 2):** a função $C_{FN}\cdot\text{FNR} + C_{FP}\cdot\text{FPR}$ equivale a supor 50% de ovos trincados. *Solução (Exp 3):* $\pi\cdot C_{FN}\cdot\text{FNR} + (1-\pi)\cdot C_{FP}\cdot\text{FPR}$ com prevalência real ($\pi \approx$ 1–5%).
10. **Afirmações sem base na documentação:** o Exp 2 atribuiu o FPR maior de 08/Nov à "iluminação noturna inclinada" (câmera e luz eram fixas) e apresentou o custo assimétrico $C_{FN} > C_{FP}$ como fato. *Regra:* condições de captura e parâmetros de negócio só entram na documentação se confirmados pela autora; custos são tratados como cenários.

### ✅ O que FUNCIONOU até o momento:
1. **Isolamento do ovo por diferença espectral ($R - B$) com rejeição de borda e recorte na resolução nativa (`locate_and_crop_egg_v2`):** Separou com 100% de sucesso os 732 ovos da Coleta v2 e preservou microtrincas finas independente da distância da câmera.
2. **Prevenção de Vazamento de Dados por Agrupamento de Ovo Físico (`egg_group_id` via timestamp $|\Delta t| \le 25\text{s}$) e Avaliação por Sessão (`session_date`):** Garante que fotos consecutivas do mesmo ovo rotacionado nunca se misturem entre Treino, Validação e Teste.
3. **Self-Normalized Spatial PatchCore (`MobileNetV3-Large` + `ResNet18` $28\times 28$) + `HybridCrackNet` com `EarlyStopping`:** Elevou o `AUROC` em `New sample/` de 62,73% (CNN Autoencoder puro) para **83,47% no modo One-Class estrito** (86,09% Val / 80,72% Test) e **91,09% Val / 89,19% Test no modo Híbrido Calibrado**, com reporte explícito de **Intervalos de Confiança de 95% (Wilson Score / Bootstrap)** e **Calibração de Limiar ($\tau$) por Custo Industrial ($C_{FN}:C_{FP}$)**. **⚠️ Esses números misturam trincas a lápis na avaliação — ver item 6 acima.**
4. **PatchCore leve One-Class (Exp 3):** `MobileNetV3-Small` truncada (191 mil parâmetros) + banco de 8000 patches de ovos bons com *augmentation*. Em trincas **reais**: teste 89,1% de AUROC (mediana 89,3% em 25 divisões; 81,7% dentro da sessão 04/Nov). Roda em CPU a ~22 ms/foto (rede + k-NN). Generaliza para a Coleta v1 (outra câmera, outros ovos): 97,3% de AUROC como teste externo.

---

## 🤖 6. Regras Obrigatórias para o Agente de IA

Se você é uma IA auxiliando neste projeto, cumpra rigorosamente estas 5 regras em todas as interações:

1. **Siga o roteiro e NÃO invente coisas fora do escopo:** Não crie arquivos, classes ou sistemas complexos que não foram pedidos. Faça exatamente o que foi solicitado no passo atual do experimento.
2. **Não tente detectar pintinhas ou sujeira agora:** Concentre os filtros, pré-processamentos e redes neurais na detecção de **ovos quebrados / rachaduras**.
3. **Garanta o *Early Stopping*:** Nunca inicie um loop de treinamento de rede neural sem configurar parada antecipada baseada na perda ou métrica de validação.
4. **Explique SEMPRE o que fez de forma didática e técnica:**
   * Toda resposta após escrever ou rodar código deve explicar claramente **o que foi feito**, citando os **termos técnicos**, **nomes das funções/classes** (ex: `EarlyStopping`, `PatchCore`, `cv2.findContours`, `DataLoader`) e **por que aquela técnica foi escolhida**. A autora precisa estar 100% a par de cada detalhe do que está acontecendo.
5. **Mantenha a seção de experimentos atualizada:** Ao final de cada teste de modelo, registre na tabela abaixo o modelo usado, hiperparâmetros, métricas obtidas e o que funcionou ou falhou.

---

## 📊 7. Diário de Experimentos (Atualizar a cada teste)

| Data | Modelo / Workflow | Dataset Usado | Configuração (Treino / Val / Teste + Early Stopping) | Resultados (AUC / F1 / Obs) | Status (Funcionou?) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Out/2026 | Extração Clássica (Otsu Cinza + Percentil 2%) | `Dataset/` (200 fotos v1) | Não-supervisionado sem separação de treino/teste | Falhou em 56,5% das segmentações; confundiu orientação da foto com ovo comprido. | ❌ Descartado |
| Out/2026 | Segmentação $R-B$ + Autoencoder Tabular (`MLPRegressor`) | `Dataset/` (192 fotos v1) | Treino só em normais; pintinhas e trincas misturadas | Segmentação 100%, mas pintinhas naturais e iluminação não-estática se misturaram às trincas. | ⚠️ Parcial (Motivou separar rachaduras de pintinhas) |
| Out/2026 | **Exp 1A:** `SmallConvAutoencoder` (CNN do zero em PyTorch) | `Dataset/` (181 Bons / 10 Cracked) | `Train`: 141 Bons \| `Val`: 20B+5C \| `Test`: 20B+5C \| `Batch=16`, `MaxEpochs=25` | `Val AUROC` subiu até **75,00%** (Época 22) e estabilizou. Com apenas 141 fotos não-estáticas, uma CNN treinada do zero tem capacidade limitada para isolar microtrincas finas (`cracked 3`, `4`). | ⚠️ Limitado sozinho (75% AUC) |
| Out/2026 | **Exp 1B:** `MobileNetV3-Small` (*Deep Embeddings* k-NN estilo *PatchCore*) + Descritores Morfológicos (*Top-Hat* + *Laplaciano*) | `Dataset/` (181 Bons / 10 Cracked) | Banco de Memória: 141 Bons (`Train`) \| `Val`: 20B+5C \| `Test`: 20B+5C | **Validação:** `ROC-AUC = 88,00%`, `Recall = 100%` (5/5).<br>**Teste:** `ROC-AUC = 99,00%`, `Acurácia = 92,00%`, `F1 = 83,33%`, `Recall = 100%` (5/5). Detectou **10 de 10** ovos quebrados! | ✅ **Sucesso! (Valida a linha do projeto)** |
| Nov/2026 | **Exp 2A:** `SmallConvAutoencoder` (One-Class CNN $64\times 64$) em `workflow_exp2_new_sample.py` | `New sample/` (732 fotos únicas: 621 Bons / 111 Quebrados em 4 dias/sessões e 81 grupos físicos) | Split anti-vazamento por Ovo Físico: `Train`=436B \| `Val`=120B+49Q \| `Test`=65B+62Q \| `EarlyStopping` (parou na Época 22, melhor Época 14) | **Teste:** `ROC-AUC = 62,73%` (95% CI: `[52,6%, 71,3%]`). Microtrincas finas e variação de textura/iluminação entre os 4 dias limitam um Autoencoder convolucional raso sozinho. | ⚠️ Baseline One-Class raso |
| Nov/2026 | **Exp 2B:** **One-Class `Self-Normalized Spatial PatchCore`** (`MobileNetV3-Large` + `ResNet18` $28\times 28$, $\Delta d_p = d_p - \text{mediana}(d)$) + Anisotropia (`max_elongation`) | `New sample/` (732 fotos: 621 Bons / 111 Quebrados) | `Train`: 436 Bons (0 Quebrados) \| `Val`: 120B+49Q \| `Test`: 65B+62Q | **Validação:** `ROC-AUC = 86,09%` (`[79,4%, 92,0%]`).<br>**Teste:** `ROC-AUC = 80,72%` (`[72,2%, 88,3%]`), `Recall = 82,3%` (`tau=3.2547`, Custo 5:1) a `93,5%` (`tau=2.3609`, Custo 10:1).<br>**Global (111 Quebrados):** `ROC-AUC = 83,47%` (`[78,7%, 88,3%]`). Subtrair a mediana intra-ovo neutraliza falsos positivos por porosidade da casca! | ✅ **Forte generalização One-Class** |
| Nov/2026 | **Exp 2C:** **`HybridCrackNet`** (Fusão `PatchCore` + `Deep MLP` Regularizado com `LayerNorm`/`GELU`/`Dropout` e `EarlyStopping`) + Limiar de Custo ($C_{FN}:C_{FP}$) | `New sample/` (732 fotos agrupadas em 81 ovos físicos sem vazamento) | `Train`: 436B+49Q \| `Val`: 120B+26Q \| `Test`: 65B+36Q \| `EarlyStopping` (parou na Época 38, melhor Época 20) | **Validação:** `ROC-AUC = 91,09%` (`[84,2%, 96,5%]`).<br>**Teste:** `ROC-AUC = 89,19%` (`[81,7%, 95,3%]`).<br>• Limiar Custo 1:1 (`tau=0.4430`): `FPR = 7,7%`, `Recall = 66,7%`, `F1 = 73,8%`<br>• Limiar Custo 5:1 (`tau=0.3509`): `Recall = 83,3%`, `FPR = 27,7%`<br>• Limiar Custo 10:1 (`tau=0.2549`): `Recall = 97,2%` (`35/36` trincados!), `FPR = 67,7%`. | ✅ **Melhor Modelo (Pronto para Revisão)** |
| Nov/2026 | **Exp 3:** **PatchCore leve One-Class** (`MobileNetV3-Small` truncada no bloco 8, congelada, 191 mil parâmetros; banco k-center de 8000 patches) vs. **Híbrido** (+ Regressão Logística com 9 features, treinada com trincas a lápis) em `workflow_exp3_real_cracks.py` | `New sample/` (732 fotos: 621 bons / 24 trincas **reais** em 7 ovos / 87 trincas **a lápis** em 17 ovos) | Divisão por ovo físico, 0 vazamento: `Train`=343B+69 lápis (×5 com *augmentation*) \| `Val`=140B+12 reais (3 ovos) \| `Test`=138B+12 reais (4 ovos) \| `Pencil_eval`=18 lápis (5 ovos). Sem treino iterativo de rede (backbone congelada + regressão logística), logo sem *Early Stopping*. Limiar pelo custo esperado com prevalência real ($\pi$ = 1/3/5/12%). 25 divisões aleatórias. | **One-Class, teste (trincas reais):** `ROC-AUC = 89,1%` (IC por ovo `[75,2%, 94,7%]`); mediana em 25 divisões **89,3%** (P10–P90 86,8–91,7%); só 04/Nov **81,7%**. **Híbrido:** 74,6% (mediana 79,5%) — lápis piorou.<br>• **Teste externo Coleta v1** (181B + 10 trincas reais, outros ovos, câmera não fixa, nunca usada em treino/limiar): One-Class **97,3%** (recall 10/10, FPR 15,5% no limiar 20:1); trincas da v1 são mais graves.<br>• $\pi$=3%, 20:1: `Recall = 75%`, `FPR = 12,3%`, prejuízo −55% vs. sem inspeção<br>• $\pi$=3%, 1:1: melhor não rejeitar nada.<br>CPU: 22 ms/foto (1 núcleo). | ✅ **Melhor modelo atual** (poucas trincas reais: IC largo) |
