import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

# Fixar sementes para reprodutibilidade científica
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def locate_and_crop_egg(image_path, output_size=128):
    """
    1. IDENTIFICADOR DO OVO (ROI Localization):
    Localiza o ovo na imagem via separação espectral (R - B) + Limiar de Otsu,
    isola a máscara do ovo e recorta a região de interesse (Bounding Box) redimensionada.
    """
    with open(image_path, "rb") as stream:
        bytes_array = bytearray(stream.read())
    np_array = np.asarray(bytes_array, dtype=np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img is None:
        return None, None

    h, w = img.shape[:2]
    scale = 600.0 / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    b_chan, g_chan, r_chan = cv2.split(resized)
    color_diff = cv2.subtract(r_chan, b_chan)
    _, thresh = cv2.threshold(color_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    best_c = max(contours, key=cv2.contourArea)
    egg_mask = np.zeros_like(mask)
    cv2.drawContours(egg_mask, [best_c], -1, 255, -1)

    # Extrair features específicas de rachadura (ignorando pintinhas escuras):
    # - Membrana branca exposta (alta luminância e baixa saturação dentro/borda do ovo)
    # - Deformação de solidez da casca
    area = float(cv2.contourArea(best_c))
    hull = cv2.convexHull(best_c)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 1.0

    # Dilatar levemente a máscara para capturar casca quebrada esbranquiçada na borda
    dilated_mask = cv2.dilate(egg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Membrana interna exposta pela quebra é clara/esbranquiçada (S baixo, V alto),
    # mas diferente do reflexo especular central (que é um círculo perfeito no topo)
    eroded_mask = cv2.erode(egg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    eroded_area = float(np.sum(eroded_mask > 0)) if np.sum(eroded_mask > 0) > 0 else 1.0

    # Filtro Morfológico Top-Hat (destaca estruturas claras finas: membrana exposta na rachadura)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    # Filtro Laplaciano para linhas abruptas de fratura
    laplacian = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F))

    white_membrane_ratio = float(np.sum((tophat > 18) & (dilated_mask > 0)) / area) if area > 0 else 0.0
    laplacian_energy = float(np.mean(laplacian[eroded_mask > 0])) if np.sum(eroded_mask > 0) > 0 else 0.0

    # Recorte (Crop) do Bounding Box do ovo com fundo mascarado em preto
    masked_img = cv2.bitwise_and(resized, resized, mask=dilated_mask)
    x, y, wb, hb = cv2.boundingRect(dilated_mask)
    crop = masked_img[y:y+hb, x:x+wb]
    crop_rgb = cv2.cvtColor(cv2.resize(crop, (output_size, output_size)), cv2.COLOR_BGR2RGB)

    crack_descriptors = np.array([
        1.0 - solidity,          # Perda de solidez por casca quebrada na borda
        white_membrane_ratio,    # Exposição de membrana branca/calcária na fratura
        laplacian_energy         # Energia de borda abrupta (Laplaciano)
    ], dtype=np.float32)

    return crop_rgb, crack_descriptors


class EggImageDataset(Dataset):
    def __init__(self, crops, descriptors, labels, filenames):
        self.crops = crops
        self.descriptors = descriptors
        self.labels = labels
        self.filenames = filenames
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Converte [0, 255] HWC para [0.0, 1.0] CHW
        ])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        tensor_img = self.transform(self.crops[idx])
        desc = torch.tensor(self.descriptors[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_img, desc, label, self.filenames[idx]


class SmallConvAutoencoder(nn.Module):
    """
    Rede Neural Convolucional Pequena (Autoencoder One-Class):
    Treinada APENAS em ovos bons para reconstruir a aparência normal da casca.
    """
    def __init__(self):
        super().__init__()
        # Encoder: 3x128x128 -> 16x64x64 -> 32x32x32 -> 64x16x16
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        # Decoder: 64x16x16 -> 32x32x32 -> 16x64x64 -> 3x128x128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def extract_deep_embeddings(crops, device):
    """
    Algoritmo estilo PatchCore / PaDiM (anomalib):
    Usa uma CNN pequena pré-treinada (MobileNetV3-Small) como extrator de características profundas
    e combina com descritores morfológicos de fratura.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    backbone = models.mobilenet_v3_small(weights=weights).to(device)
    backbone.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    embeddings = []
    with torch.no_grad():
        for crop in crops:
            inp = preprocess(crop).unsqueeze(0).to(device)
            feat = backbone.features(inp)
            pooled = torch.mean(feat, dim=[2, 3]).squeeze(0).cpu().numpy()
            embeddings.append(pooled)
    return np.array(embeddings)


def main():
    dataset_dir = 'Dataset'
    all_files = sorted([f for f in os.listdir(dataset_dir) if f.lower().endswith('.jpg')])

    good_files = [f for f in all_files if not f.lower().startswith('cracked')]
    cracked_files = sorted(
        [f for f in all_files if f.lower().startswith('cracked')],
        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
    )

    print("=========================================================================")
    print("  EXPERIMENTO 1: DETECTOR DE OVOS QUEBRADOS (ONE-CLASS NEURAL NETWORK)")
    print("=========================================================================")
    print(f"Dispositivo de execução (Hardware): {DEVICE}")
    print(f"Total de imagens encontradas      : {len(all_files)}")
    print(f"  • Ovos Bons (Normais)           : {len(good_files)}")
    print(f"  • Ovos Quebrados (Cracked 1-10) : {len(cracked_files)} -> {cracked_files}\n")

    # 1. Localizar e recortar todos os ovos
    print("[Etapa 1] Executando Identificador do Ovo (ROI Crop + Extração Morfológica)...")
    data_records = []
    for f in good_files:
        crop, desc = locate_and_crop_egg(os.path.join(dataset_dir, f))
        if crop is not None:
            data_records.append((f, crop, desc, 0))

    for f in cracked_files:
        crop, desc = locate_and_crop_egg(os.path.join(dataset_dir, f))
        if crop is not None:
            data_records.append((f, crop, desc, 1))

    # 2. Divisão em Sets conforme SPECS.md:
    # - Train Set: APENAS ovos bons (141 ovos bons)
    # - Val Set  : 20 ovos bons + 5 ovos quebrados (cracked 1 a 5)
    # - Test Set : 20 ovos bons + 5 ovos quebrados (cracked 6 a 10)
    good_records = [r for r in data_records if r[3] == 0]
    cracked_records = [r for r in data_records if r[3] == 1]

    np.random.shuffle(good_records)
    train_records = good_records[:141]
    val_records = good_records[141:161] + cracked_records[:5]
    test_records = good_records[161:181] + cracked_records[5:]

    print(f"[Etapa 2] Divisão dos Conjuntos (Sets):")
    print(f"  • Train Set (Treinamento) : {len(train_records)} imagens (100% Ovos Bons / 0 Quebrados)")
    print(f"  • Val Set   (Validação)   : {len(val_records)} imagens ({len(good_records[141:161])} Bons + 5 Quebrados: {[r[0] for r in cracked_records[:5]]})")
    print(f"  • Test Set  (Teste Final) : {len(test_records)} imagens ({len(good_records[161:181])} Bons + 5 Quebrados: {[r[0] for r in cracked_records[5:]]})\n")

    def make_loader(records, batch_size=16, shuffle=False):
        ds = EggImageDataset(
            [r[1] for r in records],
            [r[2] for r in records],
            [r[3] for r in records],
            [r[0] for r in records]
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_records, batch_size=16, shuffle=True)
    val_loader = make_loader(val_records, batch_size=16, shuffle=False)
    test_loader = make_loader(test_records, batch_size=16, shuffle=False)

    # 3. Treinamento do Autoencoder Convolucional com Early Stopping
    print("[Etapa 3] Treinando SmallConvAutoencoder (Max 25 épocas | Batch Size = 16 | Early Stopping Patience = 5)...")
    model = SmallConvAutoencoder().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    max_epochs = 25
    patience = 5
    best_val_auc = 0.0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for imgs, _, _, _ in train_loader:
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()
            recon = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validação nos ovos bons e quebrados do Val Set para monitorar AUROC com Early Stopping
        model.eval()
        val_normal_losses = []
        val_scores = []
        val_labels = []
        with torch.no_grad():
            for imgs, descs, labels, _ in val_loader:
                imgs = imgs.to(DEVICE)
                recon = model(imgs)
                # Erro de altas frequências (diferença nos top 5% pixels mais discrepantes da casca)
                diff_map = torch.mean((imgs - recon) ** 2, dim=1) # [B, H, W]
                flat_diff = diff_map.view(diff_map.size(0), -1)
                topk_vals, _ = torch.topk(flat_diff, k=int(flat_diff.size(1) * 0.05), dim=1)
                batch_scores = torch.mean(topk_vals, dim=1).cpu().numpy()

                for s, l in zip(batch_scores, labels.numpy()):
                    val_scores.append(s)
                    val_labels.append(l)
                    if l == 0:
                        val_normal_losses.append(s)

        mean_train_loss = np.mean(train_losses)
        mean_val_loss = np.mean(val_normal_losses)
        val_auc = roc_auc_score(val_labels, val_scores)

        print(f"  Época {epoch:02d}/{max_epochs:02d} | Train MSE: {mean_train_loss:.5f} | Val Normal Error: {mean_val_loss:.5f} | Val AUROC: {val_auc*100:.2f}%")

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            epochs_no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  >>> [Early Stopping Acionado na Época {epoch}] Val AUROC não melhorou por {patience} épocas consecutivas (Melhor: {best_val_auc*100:.2f}%)!")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # 4. Comparação com Deep Feature Representation (MobileNetV3-Small Backbone + k-NN Anomaly Score estilo PatchCore)
    print("\n[Etapa 4] Avaliando Abordagem Híbrida (Deep Features MobileNetV3-Small + Descritores de Fratura)...")
    train_emb = extract_deep_embeddings([r[1] for r in train_records], DEVICE)
    train_desc = np.array([r[2] for r in train_records])

    # Normalizar features usando apenas estatísticas do Train Set (Ovos Bons)
    desc_mean = train_desc.mean(axis=0)
    desc_std = train_desc.std(axis=0) + 1e-6
    train_desc_norm = (train_desc - desc_mean) / desc_std

    def compute_anomaly_scores(records):
        emb = extract_deep_embeddings([r[1] for r in records], DEVICE)
        desc = np.array([r[2] for r in records])
        desc_norm = (desc - desc_mean) / desc_std

        # Distância k-NN no espaço de Deep Embeddings (MobileNetV3) em relação ao banco de memória de ovos bons
        dists = np.linalg.norm(emb[:, None, :] - train_emb[None, :, :], axis=2)
        knn_dist = np.mean(np.sort(dists, axis=1)[:, :5], axis=1)

        # Score de fratura estrutural (perda de solidez + membrana branca exposta + Laplaciano)
        fracture_score = np.max(desc_norm, axis=1)

        # Score combinado
        combined_score = (knn_dist / (np.mean(knn_dist) + 1e-6)) + 0.5 * fracture_score
        return knn_dist, fracture_score, combined_score

    val_knn, val_frac, val_comb = compute_anomaly_scores(val_records)
    test_knn, test_frac, test_comb = compute_anomaly_scores(test_records)

    val_y = np.array([r[3] for r in val_records])
    test_y = np.array([r[3] for r in test_records])

    val_auc_comb = roc_auc_score(val_y, val_comb)
    test_auc_comb = roc_auc_score(test_y, test_comb)

    # Calibrar limiar ótimo (Threshold) no Val Set para maximizar F1-Score
    best_thresh = np.percentile(val_comb[val_y == 0], 90)
    best_f1 = -1
    for t in np.linspace(val_comb.min(), val_comb.max(), 100):
        f1_v = f1_score(val_y, (val_comb >= t).astype(int), zero_division=0)
        if f1_v > best_f1:
            best_f1 = f1_v
            best_thresh = t

    val_preds = (val_comb >= best_thresh).astype(int)
    test_preds = (test_comb >= best_thresh).astype(int)

    print("\n=========================================================================")
    print("  RESULTADOS CONSOLIDADOS DO EXPERIMENTO 1")
    print("=========================================================================")
    print(f"1. Conjunto de Validação (20 Bons + 5 Quebrados):")
    print(f"   • ROC-AUC  : {val_auc_comb*100:.2f}%")
    print(f"   • Acurácia : { np.mean(val_preds == val_y)*100:.2f}%")
    print(f"   • F1-Score : {f1_score(val_y, val_preds, zero_division=0)*100:.2f}%")
    print(f"   • Recall   : {recall_score(val_y, val_preds, zero_division=0)*100:.2f}%\n")

    print(f"2. Conjunto de Teste (20 Bons + 5 Quebrados):")
    print(f"   • ROC-AUC  : {test_auc_comb*100:.2f}%")
    print(f"   • Acurácia : {np.mean(test_preds == test_y)*100:.2f}%")
    print(f"   • F1-Score : {f1_score(test_y, test_preds, zero_division=0)*100:.2f}%")
    print(f"   • Recall   : {recall_score(test_y, test_preds, zero_division=0)*100:.2f}%\n")

    # Detalhamento individual dos 10 ovos quebrados
    all_eval_records = val_records + test_records
    all_eval_comb = np.concatenate([val_comb, test_comb])
    all_eval_knn = np.concatenate([val_knn, test_knn])
    all_eval_frac = np.concatenate([val_frac, test_frac])
    all_eval_preds = (all_eval_comb >= best_thresh).astype(int)
    all_eval_sets = ['VAL']*len(val_records) + ['TEST']*len(test_records)

    print("--- Diagnóstico Individual dos 10 Ovos Rotulados como 'Cracked' ---")
    print(f"Limiar de Corte (Threshold calibrado na Validação): {best_thresh:.3f}")
    report_rows = []
    for rec, s_Set, sc, knn_s, fr_s, pr in zip(all_eval_records, all_eval_sets, all_eval_comb, all_eval_knn, all_eval_frac, all_eval_preds):
        fn, _, desc, lbl = rec
        report_rows.append({
            'filename': fn,
            'set': s_Set,
            'true_label': 'Cracked' if lbl == 1 else 'Good',
            'pred_label': 'Cracked' if pr == 1 else 'Good',
            'correct': bool(lbl == pr),
            'anomaly_score': float(sc),
            'deep_knn_distance': float(knn_s),
            'fracture_zscore': float(fr_s),
            'solidity_loss': float(desc[0]),
            'white_membrane_ratio': float(desc[1]),
            'laplacian_energy': float(desc[2])
        })
        if lbl == 1:
            status = "DETECTADO (Verdadeiro Positivo)" if pr == 1 else "NAO DETECTADO (Falso Negativo)"
            print(f"  • [{s_Set:4s}] {fn:14s} | Score={sc:6.3f} (DeepKNN={knn_s:.2f}, FraturaZ={fr_s:+.2f}) -> {status}")

    pd.DataFrame(report_rows).to_csv('exp1_crack_results.csv', index=False)
    print("\nRelatório tabular salvo em 'exp1_crack_results.csv'.")


if __name__ == '__main__':
    main()
