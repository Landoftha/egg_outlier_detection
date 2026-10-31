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
   * *Solução definida para o novo Dataset (Coleta v2):* As fotos serão refeitas de **maneira estática** (câmera e iluminação fixas) e **girando o ovo em diferentes ângulos**, para que o modelo entenda que um ovo visto de lado não é um ovo defeituoso.
3. **Misturar pintinhas/sujeira com rachaduras no mesmo detector:**
   * *Problema:* Filtros de borda (como Canny) e desvios de cor ativavam fortemente nas pintinhas naturais da casca marrom, gerando falsos positivos de trinca.
   * *Solução definida:* Isolar o problema — focar agora 100% em rachaduras/quebras e deixar pintinhas/sujeira para outro modelo futuro.
4. **Limiarização simples (Otsu em escala de cinza) em fundo misto:**
   * *Problema:* Capturava o mousepad azul e a toalha da mesa em vez do ovo.

### ✅ O que FUNCIONOU até o momento:
1. **Isolamento do ovo por diferença espectral ($R - B$):** Separou com 100% de sucesso os ovos marrons do fundo azul.
2. **Paradigma de Reconstrução (Autoencoder preliminar em features):** Mostrou que treinar apenas com ovos normais gera um erro de reconstrução muito maior nos ovos anômalos, validando a ideia de usar modelos *One-Class* (como os da `anomalib`) diretamente nas imagens.

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
