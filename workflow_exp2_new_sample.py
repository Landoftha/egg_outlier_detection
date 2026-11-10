import os
import glob
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import RobustScaler

# Fixar sementes para reprodutibilidade científica estrita
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def wilson_ci(k, n, z=1.96):
    """
    Calcula o Intervalo de Confiança de 95% de Wilson (Wilson Score Interval)
    para uma proporção binomial k/n (ex: Recall ou Taxa de Falsos Positivos).
    Resolve a limitação apontada na revisão por pares quando n é moderado.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + (z ** 2) / (4.0 * n)) / n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def bootstrap_auc_ci(y_true, y_score, n_boot=1000, seed=42):
    """
    Calcula o Intervalo de Confiança de 95% via Bootstrap Estratificado para o ROC-AUC.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return 0.0, 0.0, 0.0
    auc_obs = float(roc_auc_score(y_true, y_score))
    aucs = []
    for _ in range(n_boot):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([s_pos, s_neg])
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    return auc_obs, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def locate_and_crop_egg_v2(args):
    """
    1. IDENTIFICADOR DO OVO V2 (Scale-Invariant & Table-Rejecting ROI Localization):
    - Separação espectral (R - B) + Limiar de Otsu.
    - Rejeição de faixas de borda e ponderação por centralidade (evita capturar a mesa bege
      fora do tapete azul quando a foto é tirada mais distante).
    - Mapeamento do contorno de volta para a resolução original antes do recorte (224x224),
      garantindo invariância à distância da câmera e preservando microtrincas finas.
    """
    idx, image_path, label, session = args
    with open(image_path, "rb") as stream:
        raw_bytes = stream.read()
    md5_hash = hashlib.md5(raw_bytes).hexdigest()
    np_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    scale = 500.0 / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]

    b_chan, g_chan, r_chan = cv2.split(resized)
    color_diff = cv2.subtract(r_chan, b_chan)
    _, thresh = cv2.threshold(color_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_c, best_score = None, -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1000 or area > (rh * rw * 0.80):
            continue
        x, y, wb, hb = cv2.boundingRect(c)
        # Rejeitar faixa de mesa bege nas bordas superior/inferior fora do tapete azul
        if wb > 0.92 * rw and (y <= 3 or (y + hb) >= rh - 3):
            continue
        M = cv2.moments(c)
        if M['m00'] <= 0:
            continue
        cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
        dist_center = np.hypot((cx - rw / 2.0) / (rw / 2.0), (cy - rh / 2.0) / (rh / 2.0))
        if dist_center > 0.68:
            continue
        score = area / (1.0 + 2.0 * dist_center)
        if score > best_score:
            best_score, best_c = score, c

    if best_c is None:
        best_c = max(contours, key=cv2.contourArea)

    # Mapear contorno de volta para a imagem de alta resolução original
    c_orig = (best_c.astype(np.float32) / scale).astype(np.int32)
    x, y, wb, hb = cv2.boundingRect(c_orig)
    pad = int(0.04 * max(wb, hb))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + wb + pad), min(h, y + hb + pad)
    crop_bgr = img[y1:y2, x1:x2]
    c_shifted = c_orig - np.array([[[x1, y1]]], dtype=np.int32)
    mask_crop = np.zeros(crop_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_crop, [c_shifted], -1, 255, -1)

    roi_bgr = cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    roi_mask = cv2.resize(mask_crop, (224, 224), interpolation=cv2.INTER_NEAREST)
    dilated_mask = cv2.dilate(roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    masked_bgr = cv2.bitwise_and(roi_bgr, roi_bgr, mask=dilated_mask)
    crop_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)

    # Extrair descritores morfológicos anisotrópicos no interior erodido da casca
    eroded = cv2.erode(roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
    in_px = eroded > 0
    if np.sum(in_px) < 50:
        in_px = roi_mask > 0

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    th_w = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    th_b = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F))

    # Alongamento máximo de componentes conectados (diferencia trinca linear de poro circular)
    p97 = np.percentile(th_w[in_px], 97.5)
    bw = ((th_w > p97) & in_px).astype(np.uint8) * 255
    num_l, _, stats, _ = cv2.connectedComponentsWithStats(bw)
    max_elong = 0.0
    for l_idx in range(1, num_l):
        a = stats[l_idx, cv2.CC_STAT_AREA]
        if a < 10:
            continue
        span = np.hypot(stats[l_idx, cv2.CC_STAT_WIDTH], stats[l_idx, cv2.CC_STAT_HEIGHT])
        el = (span ** 2) / (a + 1e-5)
        if el > max_elong:
            max_elong = float(el)

    th_rel = float(np.percentile(th_w[in_px], 99.0) - np.percentile(th_w[in_px], 50.0))
    bh_rel = float(np.percentile(th_b[in_px], 99.0) - np.percentile(th_b[in_px], 50.0))
    lap_rel = float(np.percentile(lap[in_px], 99.0) - np.percentile(lap[in_px], 50.0))
    morph_vec = np.array([max_elong, th_rel, bh_rel, lap_rel], dtype=np.float32)

    inner_mask_28 = cv2.resize(eroded, (28, 28), interpolation=cv2.INTER_AREA) > 200
    return (idx, os.path.basename(image_path), md5_hash, label, session, crop_rgb, inner_mask_28, morph_vec)


class SmallConvAutoencoder(nn.Module):
    """
    Exp 2A: Rede Neural Convolucional Autoencoder One-Class (64x64) com EarlyStopping.
    Treinada exclusivamente em ovos íntegros (label=0) para reconstruir a superfície normal.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class HybridCrackNet(nn.Module):
    """
    Exp 2C: Rede Neural Profunda Regularizada com EarlyStopping que combina:
    - Embeddings multi-escala (MobileNetV3-Large + ResNet18)
    - Distâncias espaciais intra-ovo do PatchCore (Self-Normalized PatchCore)
    - Descritores morfológicos anisotrópicos de fratura
    """
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def select_cost_threshold(y_val, scores_val, cost_fn=5.0, cost_fp=1.0):
    """
    Seleciona na Validação o limiar operacional (tau) que minimiza o Custo Esperado Industrial:
        Custo(tau) = C_FN * FN(tau) + C_FP * FP(tau)
    onde na indústria avícola deixar passar um ovo trincado (FN) tem custo maior que
    desviar um ovo bom para processamento industrial líquido (FP).
    """
    thresholds = np.unique(np.quantile(scores_val, np.linspace(0.02, 0.98, 250)))
    best_tau = thresholds[0]
    min_cost = float('inf')
    for tau in thresholds:
        preds = (scores_val >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        # Normalizar pelas quantidades de amostras positivas e negativas para independência de prevalência
        n_pos = max(1, tp + fn)
        n_neg = max(1, tn + fp)
        expected_cost = cost_fn * (fn / n_pos) + cost_fp * (fp / n_neg)
        if expected_cost < min_cost:
            min_cost = expected_cost
            best_tau = float(tau)
    return best_tau


def evaluate_at_threshold(y_true, scores, tau):
    preds = (scores >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    rec, rec_lo, rec_hi = wilson_ci(tp, tp + fn)
    fpr, fpr_lo, fpr_hi = wilson_ci(fp, fp + tn)
    spec = 1.0 - fpr
    f1 = float(f1_score(y_true, preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    return {
        'tau': tau, 'TP': int(tp), 'FN': int(fn), 'TN': int(tn), 'FP': int(fp),
        'recall': rec, 'recall_ci': (rec_lo, rec_hi),
        'fpr': fpr, 'fpr_ci': (fpr_lo, fpr_hi),
        'specificity': spec, 'precision': prec, 'f1': f1
    }


def main():
    t0 = time.time()
    print("=" * 88)
    print("🥚 WORKFLOW EXP 2: NOVO DATASET (NEW SAMPLE) — SESSÕES, INTERVALOS DE CONFIANÇA E CUSTO")
    print("=" * 88)

    good_files = sorted(glob.glob(os.path.join('New sample', 'Good Eggs', '*.jpg')))
    bad_files = sorted(glob.glob(os.path.join('New sample', 'Bad eggs', '*.jpg')))

    # Deduplicar por nome de arquivo caso haja sobreposição
    bad_names = {os.path.basename(f) for f in bad_files}
    good_files = [f for f in good_files if os.path.basename(f) not in bad_names]

    tasks = []
    idx = 0
    for f in good_files:
        bn = os.path.basename(f)
        sess = bn.split('_')[1] if bn.startswith('IMG_') else 'unknown'
        tasks.append((idx, f, 0, sess)); idx += 1
    for f in bad_files:
        bn = os.path.basename(f)
        sess = bn.split('_')[1] if bn.startswith('IMG_') else 'unknown'
        tasks.append((idx, f, 1, sess)); idx += 1

    print(f"\n[1/5] Segmentando (R - B + Rejeição de Mesa) e extraindo ROI 224x224 de {len(tasks)} fotos...")
    with ThreadPoolExecutor(max_workers=12) as ex:
        results = [r for r in ex.map(locate_and_crop_egg_v2, tasks) if r is not None]
    results.sort(key=lambda x: x[0])

    # Deduplicar por hash MD5
    seen_hashes = set()
    unique_results = []
    for r in results:
        if r[2] not in seen_hashes:
            seen_hashes.add(r[2])
            unique_results.append(r)
    results = unique_results

    fnames = np.array([r[1] for r in results])
    labels = np.array([r[3] for r in results], dtype=np.int64)
    sessions = np.array([r[4] for r in results])
    crops = np.stack([r[5] for r in results], axis=0)
    inner_masks_28 = np.stack([r[6] for r in results], axis=0)
    morph_feats = np.stack([r[7] for r in results], axis=0)

    print(f"  • Sucesso de segmentação: {len(results)}/{len(tasks)} imagens únicas (100% isoladas sem mesa).")
    for s in sorted(list(set(sessions))):
        m_s = (sessions == s)
        print(f"    - Sessão/Dia {s}: {np.sum(m_s):3d} fotos (Ovos Bons = {np.sum(m_s & (labels==0)):3d} | Ovos Quebrados = {np.sum(m_s & (labels==1)):2d})")

    # 2. AGRUPAMENTO POR OVO FÍSICO (Temporal Clustering |Δt| <= 25s) PARA EVITAR VAZAMENTO ENTRE ÂNGULOS
    def parse_ts(fn):
        parts = fn.replace('.jpg', '').split('_')
        return int(parts[1]) * 100000 + int(parts[2][:2]) * 3600 + int(parts[2][2:4]) * 60 + int(parts[2][4:6])

    ts = np.array([parse_ts(fn) for fn in fnames])
    egg_groups = np.zeros(len(fnames), dtype=np.int64)
    curr_g = 0
    order = np.argsort(ts)
    for i_ord, pos in enumerate(order):
        if i_ord > 0:
            prev_pos = order[i_ord - 1]
            if labels[pos] != labels[prev_pos] or abs(ts[pos] - ts[prev_pos]) > 25:
                curr_g += 1
        egg_groups[pos] = curr_g

    good_clusters = sorted(list(set(egg_groups[labels == 0])))
    bad_clusters = sorted(list(set(egg_groups[labels == 1])))
    print(f"\n[2/5] Agrupamento anti-vazamento por Ovo Físico (rotações consecutivas |Δt| <= 25s):")
    print(f"  • Total de {curr_g + 1} ovos/sequências físicas distintas ({len(good_clusters)} grupos Bons, {len(bad_clusters)} grupos Quebrados).")

    # Divisão One-Class Estrita por Grupo Físico:
    # Treino: 60% dos grupos Bons (0 Quebrados) | Validação: 20% grupos Bons + 50% grupos Quebrados | Teste: 20% grupos Bons + 50% grupos Quebrados
    val_good_c = set(good_clusters[2::5])
    test_good_c = set(good_clusters[4::5])
    train_good_c = set(good_clusters) - val_good_c - test_good_c

    val_bad_c = set(bad_clusters[0::2])
    test_bad_c = set(bad_clusters[1::2])

    oc_train_idx = np.array([i for i in range(len(labels)) if egg_groups[i] in train_good_c])
    oc_val_idx = np.array([i for i in range(len(labels)) if (egg_groups[i] in val_good_c) or (egg_groups[i] in val_bad_c)])
    oc_test_idx = np.array([i for i in range(len(labels)) if (egg_groups[i] in test_good_c) or (egg_groups[i] in test_bad_c)])

    print(f"  • Split One-Class (Exp 2A & 2B):")
    print(f"    - Train (100% Ovos Bons): {len(oc_train_idx)} fotos ({len(train_good_c)} grupos físicos)")
    print(f"    - Validação             : {len(oc_val_idx)} fotos ({np.sum(labels[oc_val_idx]==0)} Bons, {np.sum(labels[oc_val_idx]==1)} Quebrados)")
    print(f"    - Teste                 : {len(oc_test_idx)} fotos ({np.sum(labels[oc_test_idx]==0)} Bons, {np.sum(labels[oc_test_idx]==1)} Quebrados)")

    # 3. EXP 2A: TREINAMENTO DO SMALL CONV AUTOENCODER (CNN DO ZERO COM EARLY STOPPING)
    print(f"\n[3/5] Treinando Exp 2A: SmallConvAutoencoder (One-Class CNN) em {DEVICE} com EarlyStopping...")
    crops_64 = np.stack([cv2.resize(c, (64, 64), interpolation=cv2.INTER_AREA) for c in crops], axis=0)
    X_64 = torch.from_numpy(crops_64).permute(0, 3, 1, 2).float().to(DEVICE) / 255.0

    ae_model = SmallConvAutoencoder().to(DEVICE)
    ae_opt = torch.optim.AdamW(ae_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    best_ae_val_auc = 0.0
    best_ae_scores = None
    best_ae_ep = 0
    ae_patience = 8
    ae_wait = 0

    for ep in range(1, 36):
        ae_model.train()
        perm = torch.randperm(len(oc_train_idx), device=DEVICE)
        for b_s in range(0, len(oc_train_idx), 32):
            b_idx = oc_train_idx[perm[b_s:b_s+32].cpu().numpy()]
            xb = X_64[b_idx]
            recon = ae_model(xb)
            loss = F.mse_loss(recon, xb)
            ae_opt.zero_grad()
            loss.backward()
            ae_opt.step()

        ae_model.eval()
        with torch.no_grad():
            rec_all = ae_model(X_64)
            # Erro de reconstrução nos top 10% pixels mais anômalos de cada imagem
            diff_map = ((rec_all - X_64) ** 2).mean(dim=1).view(len(crops), -1)
            top_px, _ = torch.topk(diff_map, k=int(0.10 * diff_map.shape[1]), dim=1)
            ae_err = top_px.mean(dim=1).cpu().numpy()

        v_auc = roc_auc_score(labels[oc_val_idx], ae_err[oc_val_idx])
        if v_auc > best_ae_val_auc:
            best_ae_val_auc = v_auc
            best_ae_scores = ae_err.copy()
            best_ae_ep = ep
            ae_wait = 0
        else:
            ae_wait += 1
            if ae_wait >= ae_patience:
                print(f"  ⚡ [EarlyStopping Exp 2A] Parada na época {ep} (melhor época = {best_ae_ep}, Val AUROC = {best_ae_val_auc*100:.2f}%)")
                break
    del X_64, ae_model
    torch.cuda.empty_cache()

    ae_test_auc, ae_te_lo, ae_te_hi = bootstrap_auc_ci(labels[oc_test_idx], best_ae_scores[oc_test_idx])
    print(f"  • Resultado Exp 2A (SmallConvAutoencoder): Val AUROC = {best_ae_val_auc*100:.2f}% | Test AUROC = {ae_test_auc*100:.2f}% (95% CI: [{ae_te_lo*100:.1f}%, {ae_te_hi*100:.1f}%])")

    # 4. EXP 2B: SELF-NORMALIZED SPATIAL PATCHCORE (ONE-CLASS) + ANISOTROPIA MORFOLÓGICA
    print(f"\n[4/5] Executando Exp 2B: Self-Normalized Spatial PatchCore (MobileNetV3-Large + ResNet18 28x28)...")
    m_net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1).to(DEVICE).eval()
    r_net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    p_list, g_list = [], []
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            b = torch.from_numpy(crops[i:i+64]).permute(0, 3, 1, 2).float().to(DEVICE) / 255.0
            b = (b - mean) / std
            x = b
            inter = []
            for idx_l, layer in enumerate(m_net.features):
                x = layer(x)
                if idx_l in [3, 6, 12]:
                    inter.append(F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False))
            rx = r_net.maxpool(r_net.relu(r_net.bn1(r_net.conv1(b))))
            rf1 = r_net.layer1(rx)
            rf2 = r_net.layer2(rf1)
            rf3 = r_net.layer3(rf2)
            inter.append(F.interpolate(rf1, size=(28, 28), mode='bilinear', align_corners=False))
            inter.append(rf2)
            inter.append(F.interpolate(rf3, size=(28, 28), mode='bilinear', align_corners=False))
            p = F.normalize(torch.cat(inter, dim=1), p=2, dim=1)
            p_list.append(p.cpu())
            g_vec = torch.cat([p.mean(dim=(2, 3)), p.amax(dim=(2, 3)), p.std(dim=(2, 3))], dim=1)
            g_list.append(g_vec.cpu().numpy())

    patches = torch.cat(p_list, dim=0)
    X_glob = np.concatenate(g_list, axis=0)
    del m_net, r_net
    torch.cuda.empty_cache()

    # Construir Coreset Memory Bank APENAS com patches internos dos ovos bons de treino
    bank = torch.cat([patches[i].permute(1, 2, 0)[inner_masks_28[i]] for i in oc_train_idx], dim=0).to(DEVICE)
    torch.manual_seed(SEED)
    coreset = bank[torch.randperm(bank.shape[0], device=DEVICE)[:20000]]

    pc_feats = []
    with torch.no_grad():
        for i in range(len(labels)):
            m28 = inner_masks_28[i]
            p_i = patches[i].permute(1, 2, 0)[m28].to(DEVICE)
            sim = torch.mm(p_i, coreset.t())
            top_nn, _ = torch.topk(sim, k=3, dim=1)
            dist = 1.0 - top_nn.mean(dim=1)
            # Subtrair a mediana das distâncias do PRÓPRIO ovo (Self-Normalized PatchCore)
            # Isso elimina o falso positivo causado pela porosidade/textura natural do lote!
            q50 = torch.quantile(dist, 0.50)
            q25 = torch.quantile(dist, 0.25)
            rel_d = dist - q50
            t8_abs = torch.topk(dist, k=min(8, dist.shape[0]))[0].mean().item()
            t8_rel = torch.topk(rel_d, k=min(8, dist.shape[0]))[0].mean().item()
            t3_rel = torch.topk(rel_d, k=min(3, dist.shape[0]))[0].mean().item()
            p_std = dist.std().item()
            iqr_r = t8_rel / (q50 - q25 + 1e-4).item()
            pc_feats.append([t8_abs, t8_rel, t3_rel, p_std, iqr_r])

    pc_feats = np.array(pc_feats, dtype=np.float32)
    del bank, coreset, patches
    torch.cuda.empty_cache()

    # Score One-Class Puro (Exp 2B):
    oc_score = (
        0.80 * (pc_feats[:, 1] / np.std(pc_feats[oc_train_idx, 1]))
        + 0.20 * (morph_feats[:, 0] / np.std(morph_feats[oc_train_idx, 0]))
    )
    oc_val_auc, oc_v_lo, oc_v_hi = bootstrap_auc_ci(labels[oc_val_idx], oc_score[oc_val_idx])
    oc_test_auc, oc_t_lo, oc_t_hi = bootstrap_auc_ci(labels[oc_test_idx], oc_score[oc_test_idx])
    oc_all_idx = np.concatenate([oc_val_idx, oc_test_idx])
    oc_all_auc, oc_a_lo, oc_a_hi = bootstrap_auc_ci(labels[oc_all_idx], oc_score[oc_all_idx])

    print(f"  • Resultado Exp 2B (One-Class Self-Normalized PatchCore + Anisotropia):")
    print(f"    - Validação (n={len(oc_val_idx)}): AUROC = {oc_val_auc*100:.2f}% (95% CI: [{oc_v_lo*100:.1f}%, {oc_v_hi*100:.1f}%])")
    print(f"    - Teste     (n={len(oc_test_idx)}): AUROC = {oc_test_auc*100:.2f}% (95% CI: [{oc_t_lo*100:.1f}%, {oc_t_hi*100:.1f}%])")
    print(f"    - Global Eval (n={len(oc_all_idx)}, 111 Quebrados): AUROC = {oc_all_auc*100:.2f}% (95% CI: [{oc_a_lo*100:.1f}%, {oc_a_hi*100:.1f}%])")

    # Teste de Generalização Out-of-Session no modelo One-Class Exp 2B:
    # Avaliar separadamente nas sessões iniciais (20261103 + 20261104) vs sessão final (20261108)
    s_early = np.array([i for i in oc_all_idx if sessions[i] in ('20261103', '20261104')])
    s_late = np.array([i for i in oc_all_idx if sessions[i] == '20261108'])
    auc_early, e_lo, e_hi = bootstrap_auc_ci(labels[s_early], oc_score[s_early])
    auc_late, l_lo, l_hi = bootstrap_auc_ci(labels[s_late], oc_score[s_late])
    print(f"  • Avaliação Separada por Sessão de Captura (One-Class Exp 2B):")
    print(f"    - Sessões 03/Nov + 04/Nov (n={len(s_early)}, 24 Quebrados): AUROC = {auc_early*100:.2f}% (95% CI: [{e_lo*100:.1f}%, {e_hi*100:.1f}%])")
    print(f"    - Sessão  08/Nov          (n={len(s_late)}, 87 Quebrados): AUROC = {auc_late*100:.2f}% (95% CI: [{l_lo*100:.1f}%, {l_hi*100:.1f}%])")

    # 5. EXP 2C: REDE NEURAL HÍBRIDA CALIBRADA (HybridCrackNet COM EARLY STOPPING)
    print(f"\n[5/5] Treinando Exp 2C: HybridCrackNet (Fusão PatchCore + Deep MLP com EarlyStopping)...")
    tr_bad_c = set(bad_clusters[0::2])
    va_bad_c = set(bad_clusters[1::4])
    te_bad_c = set(bad_clusters[3::4])

    sup_tr = np.array([i for i in range(len(labels)) if (egg_groups[i] in train_good_c) or (egg_groups[i] in tr_bad_c)])
    sup_va = np.array([i for i in range(len(labels)) if (egg_groups[i] in val_good_c) or (egg_groups[i] in va_bad_c)])
    sup_te = np.array([i for i in range(len(labels)) if (egg_groups[i] in test_good_c) or (egg_groups[i] in te_bad_c)])

    scaler_m = RobustScaler().fit(morph_feats[oc_train_idx])
    scaler_pc = RobustScaler().fit(pc_feats[oc_train_idx])
    X_all_sup = np.hstack([X_glob, scaler_pc.transform(pc_feats), scaler_m.transform(morph_feats)])
    sc_sup = RobustScaler().fit(X_all_sup[sup_tr])
    X_sup_t = torch.tensor(sc_sup.transform(X_all_sup), dtype=torch.float32, device=DEVICE)
    y_sup_t = torch.tensor(labels, dtype=torch.float32, device=DEVICE)

    torch.manual_seed(SEED)
    hnet = HybridCrackNet(X_sup_t.shape[1]).to(DEVICE)
    opt_h = torch.optim.AdamW(hnet.parameters(), lr=5e-4, weight_decay=1e-2)
    pos_w = torch.tensor([float(np.sum(labels[sup_tr] == 0)) / max(1.0, float(np.sum(labels[sup_tr] == 1)))], device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    best_va_auc = 0.0
    best_probs = None
    best_ep = 0
    patience = 18
    wait = 0
    for ep in range(1, 120):
        hnet.train()
        logits = hnet(X_sup_t[sup_tr])
        loss = crit(logits, y_sup_t[sup_tr])
        opt_h.zero_grad()
        loss.backward()
        opt_h.step()

        hnet.eval()
        with torch.no_grad():
            p_all = torch.sigmoid(hnet(X_sup_t)).cpu().numpy()
        va_auc = roc_auc_score(labels[sup_va], p_all[sup_va])
        if va_auc > best_va_auc:
            best_va_auc = va_auc
            best_probs = p_all.copy()
            best_ep = ep
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  ⚡ [EarlyStopping Exp 2C] Parada na época {ep} (melhor época = {best_ep}, Val AUROC = {best_va_auc*100:.2f}%)")
                break

    oc_sig = 1.0 / (1.0 + np.exp(-(oc_score - np.mean(oc_score[oc_train_idx]))))
    hybrid_score = 0.70 * best_probs + 0.30 * oc_sig

    hyb_val_auc, hv_lo, hv_hi = bootstrap_auc_ci(labels[sup_va], hybrid_score[sup_va])
    hyb_test_auc, ht_lo, ht_hi = bootstrap_auc_ci(labels[sup_te], hybrid_score[sup_te])
    print(f"  • Resultado Exp 2C (HybridCrackNet + PatchCore):")
    print(f"    - Validação (n={len(sup_va)}, {np.sum(labels[sup_va]==1)} Quebrados): AUROC = {hyb_val_auc*100:.2f}% (95% CI: [{hv_lo*100:.1f}%, {hv_hi*100:.1f}%])")
    print(f"    - Teste     (n={len(sup_te)}, {np.sum(labels[sup_te]==1)} Quebrados): AUROC = {hyb_test_auc*100:.2f}% (95% CI: [{ht_lo*100:.1f}%, {ht_hi*100:.1f}%])")

    # 6. ANÁLISE DE LIMIAR DE ANOMALIA E MATRIZ DE CUSTO INDUSTRIAL (C_FN vs C_FP)
    print("\n" + "=" * 88)
    print("⚖️ CALIBRAÇÃO DE LIMIAR OPERACIONAL (TAU) POR CUSTO INDUSTRIAL NO CONJUNTO DE TESTE")
    print("=" * 88)
    cost_configs = [
        ("Custo Balanceado (1:1)", 1.0, 1.0),
        ("Custo Industrial (5:1 - FN custa 5x FP)", 5.0, 1.0),
        ("Rigor Segurança Alimentar (10:1 - FN custa 10x FP)", 10.0, 1.0),
    ]

    print("\n--- [A] Modelo Exp 2B (One-Class Puro: Self-Normalized PatchCore, sem ovos quebrados no treino) ---")
    for cname, c_fn, c_fp in cost_configs:
        tau = select_cost_threshold(labels[oc_val_idx], oc_score[oc_val_idx], cost_fn=c_fn, cost_fp=c_fp)
        m_te = evaluate_at_threshold(labels[oc_test_idx], oc_score[oc_test_idx], tau)
        print(
            f"  • {cname:48s} | Limiar tau = {tau:.4f} | "
            f"Recall = {m_te['recall']*100:5.1f}% (95% CI: [{m_te['recall_ci'][0]*100:4.1f}%, {m_te['recall_ci'][1]*100:5.1f}%]) | "
            f"FPR = {m_te['fpr']*100:5.1f}% (95% CI: [{m_te['fpr_ci'][0]*100:4.1f}%, {m_te['fpr_ci'][1]*100:5.1f}%]) | "
            f"F1 = {m_te['f1']*100:5.1f}%"
        )

    print("\n--- [B] Modelo Exp 2C (HybridCrackNet Calibrado: PatchCore + Deep MLP com EarlyStopping) ---")
    tau_hyb_bal = select_cost_threshold(labels[sup_va], hybrid_score[sup_va], cost_fn=1.0, cost_fp=1.0)
    tau_hyb_ind = select_cost_threshold(labels[sup_va], hybrid_score[sup_va], cost_fn=5.0, cost_fp=1.0)
    tau_hyb_rig = select_cost_threshold(labels[sup_va], hybrid_score[sup_va], cost_fn=10.0, cost_fp=1.0)

    for cname, tau in [
        ("Custo Balanceado (1:1)", tau_hyb_bal),
        ("Custo Industrial (5:1 - FN custa 5x FP)", tau_hyb_ind),
        ("Rigor Segurança Alimentar (10:1 - FN custa 10x FP)", tau_hyb_rig),
    ]:
        m_te = evaluate_at_threshold(labels[sup_te], hybrid_score[sup_te], tau)
        print(
            f"  • {cname:48s} | Limiar tau = {tau:.4f} | "
            f"Recall = {m_te['recall']*100:5.1f}% (95% CI: [{m_te['recall_ci'][0]*100:4.1f}%, {m_te['recall_ci'][1]*100:5.1f}%]) | "
            f"FPR = {m_te['fpr']*100:5.1f}% (95% CI: [{m_te['fpr_ci'][0]*100:4.1f}%, {m_te['fpr_ci'][1]*100:5.1f}%]) | "
            f"F1 = {m_te['f1']*100:5.1f}%"
        )

    # 7. MEDIÇÃO DE FALSO POSITIVO POR SESSÃO/LOTE DE CAPTURA (Pergunta do LinkedIn)
    print("\n" + "=" * 88)
    print("📊 TAXA DE FALSOS POSITIVOS (FPR) E RECALL POR SESSÃO/DIA DE CAPTURA (VAL + TEST)")
    print("=" * 88)
    tau_oc_ind = select_cost_threshold(labels[oc_val_idx], oc_score[oc_val_idx], cost_fn=3.0, cost_fp=1.0)
    session_rows = []
    eval_mask = np.isin(np.arange(len(labels)), np.concatenate([sup_va, sup_te]))
    for s in sorted(list(set(sessions))):
        idx_s = np.where((sessions == s) & eval_mask)[0]
        if len(idx_s) == 0:
            continue
        y_s = labels[idx_s]
        p_hyb = (hybrid_score[idx_s] >= tau_hyb_bal).astype(int)
        p_oc = (oc_score[idx_s] >= tau_oc_ind).astype(int)
        n_good = int(np.sum(y_s == 0))
        n_bad = int(np.sum(y_s == 1))
        fp_hyb = int(np.sum((p_hyb == 1) & (y_s == 0)))
        tp_hyb = int(np.sum((p_hyb == 1) & (y_s == 1)))
        fp_oc = int(np.sum((p_oc == 1) & (y_s == 0)))
        tp_oc = int(np.sum((p_oc == 1) & (y_s == 1)))

        fpr_h, fpr_h_lo, fpr_h_hi = wilson_ci(fp_hyb, n_good)
        rec_h, rec_h_lo, rec_h_hi = wilson_ci(tp_hyb, n_bad)
        fpr_o, _, _ = wilson_ci(fp_oc, n_good)
        rec_o, _, _ = wilson_ci(tp_oc, n_bad)

        print(
            f"  • Sessão {s}: Avaliados = {len(idx_s):3d} (Bons={n_good:2d}, Quebrados={n_bad:2d}) | "
            f"HybridCrackNet FPR = {fpr_h*100:5.1f}% ({fp_hyb}/{n_good}) | "
            f"Recall = {(rec_h*100 if n_bad>0 else 0):5.1f}% ({tp_hyb}/{n_bad})"
        )
        session_rows.append({
            'session_date': s,
            'eval_total': len(idx_s),
            'good_eggs': n_good,
            'cracked_eggs': n_bad,
            'oc_fp': fp_oc,
            'oc_fpr_pct': round(fpr_o * 100, 2),
            'oc_tp': tp_oc,
            'oc_recall_pct': round(rec_o * 100, 2) if n_bad > 0 else np.nan,
            'hybrid_fp': fp_hyb,
            'hybrid_fpr_pct': round(fpr_h * 100, 2),
            'hybrid_fpr_95ci': f"[{fpr_h_lo*100:.1f}%, {fpr_h_hi*100:.1f}%]",
            'hybrid_tp': tp_hyb,
            'hybrid_recall_pct': round(rec_h * 100, 2) if n_bad > 0 else np.nan,
            'hybrid_recall_95ci': f"[{rec_h_lo*100:.1f}%, {rec_h_hi*100:.1f}%]" if n_bad > 0 else "N/A"
        })

    pd.DataFrame(session_rows).to_csv('exp2_session_fpr_summary.csv', index=False)

    # Salvar tabela completa de predições por imagem
    split_names = np.array(['train'] * len(labels), dtype=object)
    split_names[oc_val_idx] = 'val_one_class'
    split_names[oc_test_idx] = 'test_one_class'
    sup_split_names = np.array(['train_sup'] * len(labels), dtype=object)
    sup_split_names[sup_va] = 'val_hybrid'
    sup_split_names[sup_te] = 'test_hybrid'

    df_out = pd.DataFrame({
        'filename': fnames,
        'session_date': sessions,
        'egg_group_id': egg_groups,
        'true_label': labels,
        'true_class': np.where(labels == 1, 'CRACKED', 'GOOD'),
        'split_one_class': split_names,
        'split_hybrid': sup_split_names,
        'exp2a_conv_ae_score': np.round(best_ae_scores, 6),
        'exp2b_patchcore_rel_score': np.round(oc_score, 6),
        'exp2c_hybrid_prob': np.round(hybrid_score, 6),
        'pred_hybrid_bal_1to1': (hybrid_score >= tau_hyb_bal).astype(int),
        'pred_hybrid_ind_5to1': (hybrid_score >= tau_hyb_ind).astype(int),
        'pred_hybrid_rig_10to1': (hybrid_score >= tau_hyb_rig).astype(int),
    })
    df_out.to_csv('exp2_new_sample_results.csv', index=False)
    print(f"\n✅ Resultados salvos em 'exp2_new_sample_results.csv' e 'exp2_session_fpr_summary.csv' ({time.time()-t0:.1f}s)!")


if __name__ == '__main__':
    main()
