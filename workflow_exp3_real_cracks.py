"""
EXP 3 — Detecção de trincas com avaliação SOMENTE em trincas reais, custo ponderado pela
prevalência real e modelo leve (roda em CPU).

Mudanças em relação ao Exp 2 (workflow_exp2_new_sample.py):
  1. As trincas da sessão 08/Nov foram DESENHADAS A LÁPIS (amostras sintéticas criadas para
     contornar a falta de ovos trincados). Elas entram apenas no TREINO; uma parte é separada
     (nunca vista no treino) só para um relatório à parte. Validação e teste usam apenas
     trincas REAIS (sessões 03/Nov e 04/Nov).
  2. Divisão por OVO FÍSICO (fotos consecutivas |Δt| <= 25 s), estratificada por sessão,
     com 100% das trincas reais em validação/teste (nenhuma no treino).
  3. Data augmentation apenas no treino (rotação, espelhamento, brilho/contraste/gama,
     escala, desfoque leve).
  4. Correção de vazamento do Exp 2: as features PatchCore dos ovos de treino eram calculadas
     contra um banco de memória que continha o próprio ovo (distância ~0). Aqui elas são
     calculadas out-of-fold (banco construído sem o grupo físico do próprio ovo).
  5. Modelo leve: MobileNetV3-Small truncada (até o bloco 8, congelada) + k-NN em banco de
     memória (coreset) + regressão logística de 9 features.
  6. Limiar escolhido pelo menor CUSTO ESPERADO POR OVO usando a prevalência real de trincas:
        custo(τ) = π·C_FN·FNR(τ) + (1 − π)·C_FP·FPR(τ)
     (o Exp 2 ignorava π, equivalente a supor 50% de ovos trincados).
"""
import os
import glob
import json
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------------------
# Parâmetros de negócio (editáveis)
# ---------------------------------------------------------------------------------------
# Prevalência real de ovos trincados na produção. Literatura: ~1–5% na granja
# (HN Layers: 1–5% trincas/buracos; estudos de packing plant: 3,95% e 4,9%) e até ~12% de
# problemas de casca entre granja e consumidor (Embrapa). Cenário base = 3%.
PREVALENCE_BASE = 0.03
PREVALENCE_SCENARIOS = [0.01, 0.03, 0.05, 0.12]
# Razão de custo C_FN / C_FP. C_FP = 1 unidade (um ovo bom desviado/descartado).
# O valor real depende da granja (preço do ovo, destino do ovo trincado, multas/reclamações),
# por isso o limiar é reportado para vários cenários.
COST_RATIOS = [1, 5, 10, 20, 50]

N_AUG = 4                  # cópias aumentadas por imagem de treino (além da original)
CORESET_SIZE = 8000        # patches no banco de memória (≈ 8000 x 72 floats ≈ 2,3 MB)
FEATURE_LAYERS = (3, 8)    # blocos da MobileNetV3-Small usados (24 + 48 = 72 canais @ 28x28)
N_REPEATS = 25             # repetições com divisões aleatórias diferentes (variância)
GROUP_GAP_S = 25


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------------------
# Estatística
# ---------------------------------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    denom = 1.0 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def group_bootstrap_auc(y, s, groups, n_boot=2000, seed=SEED):
    """IC 95% do AUROC reamostrando OVOS FÍSICOS (não fotos), pois fotos do mesmo ovo são correlacionadas."""
    rng = np.random.default_rng(seed)
    auc = float(roc_auc_score(y, s))
    pos_g = np.unique(groups[y == 1])
    neg_g = np.unique(groups[y == 0])
    idx_by_g = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    aucs = []
    for _ in range(n_boot):
        gs = np.concatenate([rng.choice(pos_g, len(pos_g)), rng.choice(neg_g, len(neg_g))])
        idx = np.concatenate([idx_by_g[g] for g in gs])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return auc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ---------------------------------------------------------------------------------------
# Localização do ovo (mesma lógica validada no Exp 2) + features morfológicas
# ---------------------------------------------------------------------------------------
def locate_egg(path):
    with open(path, 'rb') as fh:
        raw = fh.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = 500.0 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rh, rw = small.shape[:2]
    b, _, r = cv2.split(small)
    _, th = cv2.threshold(cv2.subtract(r, b), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(cv2.morphologyEx(th, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best, best_sc = None, -1.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 1000 or a > rh * rw * 0.8:
            continue
        x, y, wb, hb = cv2.boundingRect(c)
        if wb > 0.92 * rw and (y <= 3 or y + hb >= rh - 3):
            continue
        M = cv2.moments(c)
        if M['m00'] <= 0:
            continue
        cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
        d = np.hypot((cx - rw / 2) / (rw / 2), (cy - rh / 2) / (rh / 2))
        if d > 0.68:
            continue
        sc = a / (1 + 2 * d)
        if sc > best_sc:
            best_sc, best = sc, c
    if best is None:
        best = max(cnts, key=cv2.contourArea)
    c0 = (best.astype(np.float32) / scale).astype(np.int32)
    x, y, wb, hb = cv2.boundingRect(c0)
    pad = int(0.04 * max(wb, hb))
    x1, y1, x2, y2 = max(0, x - pad), max(0, y - pad), min(w, x + wb + pad), min(h, y + hb + pad)
    crop = img[y1:y2, x1:x2]
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.drawContours(mask, [c0 - np.array([[[x1, y1]]], np.int32)], -1, 255, -1)
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    return hashlib.md5(raw).hexdigest(), cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), mask


def mask_and_morph(crop_rgb, mask):
    """Aplica a máscara, devolve a imagem mascarada, a máscara interna 28x28 e 4 descritores morfológicos."""
    dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    masked = cv2.bitwise_and(crop_rgb, crop_rgb, mask=dil)
    eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
    inner = eroded > 0
    if inner.sum() < 50:
        inner = mask > 0
    gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th_w = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k9)
    th_b = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k9)
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F))
    p97 = np.percentile(th_w[inner], 97.5)
    bw = ((th_w > p97) & inner).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(bw)
    elong = 0.0
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a >= 10:
            elong = max(elong, np.hypot(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) ** 2 / (a + 1e-5))
    rel = lambda arr: float(np.percentile(arr[inner], 99) - np.percentile(arr[inner], 50))
    morph = np.array([elong, rel(th_w), rel(th_b), rel(lap)], np.float32)
    inner28 = cv2.resize(eroded, (28, 28), interpolation=cv2.INTER_AREA) > 200
    if inner28.sum() < 10:
        inner28 = cv2.resize(mask, (28, 28), interpolation=cv2.INTER_AREA) > 200
    return masked, inner28, morph


def augment(crop_rgb, mask, rng):
    """Augmentation geométrica + fotométrica (só treino). A máscara acompanha a geometria."""
    ang = rng.uniform(0, 360)
    sc = rng.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((112, 112), ang, sc)
    img = cv2.warpAffine(crop_rgb, M, (224, 224), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    msk = cv2.warpAffine(mask, M, (224, 224), flags=cv2.INTER_NEAREST, borderValue=0)
    if rng.random() < 0.5:
        img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
    f = img.astype(np.float32) / 255.0
    f = np.clip((f - 0.5) * rng.uniform(0.85, 1.15) + 0.5 + rng.uniform(-0.08, 0.08), 0, 1)
    f = f ** rng.uniform(0.85, 1.15)
    img = (f * 255).astype(np.uint8)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, msk


# ---------------------------------------------------------------------------------------
# Backbone leve
# ---------------------------------------------------------------------------------------
class LightBackbone(torch.nn.Module):
    """MobileNetV3-Small pré-treinada (ImageNet), congelada e truncada no último bloco usado."""

    def __init__(self, layers=FEATURE_LAYERS):
        super().__init__()
        full = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.layers = layers
        self.body = torch.nn.Sequential(*list(full.features.children())[: max(layers) + 1]).eval()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x_uint8_nhwc):
        x = torch.as_tensor(x_uint8_nhwc).permute(0, 3, 1, 2).float().to(self.mean.device) / 255.0
        x = (x - self.mean) / self.std
        outs = []
        for i, blk in enumerate(self.body):
            x = blk(x)
            if i in self.layers:
                # suavização local (agregação de vizinhança 3x3, como no PatchCore)
                y = F.avg_pool2d(x, 3, 1, 1)
                outs.append(F.interpolate(y, size=(28, 28), mode='bilinear', align_corners=False))
        return F.normalize(torch.cat(outs, 1), dim=1)  # (N, 72, 28, 28)


def extract_patches(backbone, imgs, inner_masks, bs=64):
    out = []
    for i in range(0, len(imgs), bs):
        p = backbone(np.stack(imgs[i:i + bs])).permute(0, 2, 3, 1).cpu()
        for j in range(p.shape[0]):
            out.append(p[j][torch.from_numpy(inner_masks[i + j])].half())
    return out


def greedy_coreset(feats, n, seed=SEED):
    """k-center greedy (PatchCore) com projeção aleatória para 32-d."""
    if feats.shape[0] <= n:
        return feats
    g = torch.Generator(device=feats.device).manual_seed(seed)
    proj = torch.randn(feats.shape[1], 32, device=feats.device, generator=g)
    z = feats @ proj
    sel = [int(torch.randint(0, z.shape[0], (1,), generator=g, device=feats.device))]
    dmin = torch.cdist(z, z[sel[0]:sel[0] + 1]).squeeze(1)
    for _ in range(n - 1):
        k = int(torch.argmax(dmin))
        sel.append(k)
        dmin = torch.minimum(dmin, torch.cdist(z, z[k:k + 1]).squeeze(1))
    return feats[sel]


def build_bank(patch_lists, seed=SEED, greedy=True):
    """greedy=True: k-center greedy (divisão principal). greedy=False: amostragem aleatória (repetições, ~100x mais rápido)."""
    feats = torch.cat(patch_lists, 0).float().to(DEVICE)
    if greedy:
        return greedy_coreset(feats, CORESET_SIZE, seed)
    g = torch.Generator(device=feats.device).manual_seed(seed)
    return feats[torch.randperm(feats.shape[0], generator=g, device=feats.device)[:CORESET_SIZE]]


def patch_scores(patch_lists, bank, k=3):
    """5 estatísticas por imagem da distância k-NN dos patches ao banco (com auto-normalização pela mediana)."""
    out = []
    for p in patch_lists:
        p = p.float().to(DEVICE)
        sim = torch.topk(p @ bank.T, k, dim=1).values.mean(1)
        d = 1.0 - sim
        q50, q25 = torch.quantile(d, 0.5), torch.quantile(d, 0.25)
        rel = d - q50
        t8 = min(8, d.shape[0])
        t8_rel = torch.topk(rel, t8).values.mean().item()
        out.append([torch.topk(d, t8).values.mean().item(), t8_rel,
                    torch.topk(rel, min(3, d.shape[0])).values.mean().item(),
                    d.std().item(), t8_rel / (q50 - q25 + 1e-4).item()])
    return np.array(out, np.float32)


# ---------------------------------------------------------------------------------------
# Limiar por custo esperado com prevalência real
# ---------------------------------------------------------------------------------------
def expected_cost(fnr, fpr, prev, ratio):
    """Custo esperado por ovo, em unidades de C_FP (= valor de um ovo bom desviado)."""
    return prev * ratio * fnr + (1 - prev) * fpr


def pick_threshold(y, s, prev, ratio):
    cand = np.concatenate([[-np.inf], np.unique(s), [np.inf]])
    best_t, best_c = None, np.inf
    for t in cand:
        pred = s >= t
        fnr = np.mean(~pred[y == 1])
        fpr = np.mean(pred[y == 0])
        c = expected_cost(fnr, fpr, prev, ratio)
        if c < best_c - 1e-12:
            best_c, best_t = c, t
    # Ponto médio até o próximo score para não ficar "colado" numa amostra de validação
    higher = s[s >= best_t] if np.isfinite(best_t) else np.array([])
    lower = s[s < best_t] if np.isfinite(best_t) else np.array([])
    if len(higher) and len(lower):
        best_t = 0.5 * (higher.min() + lower.max())
    return float(best_t)


def rates(y, s, t):
    pred = s >= t
    tn, fp, fn, tp = confusion_matrix(y, pred.astype(int), labels=[0, 1]).ravel()
    return int(tp), int(fn), int(tn), int(fp)


# ---------------------------------------------------------------------------------------
# Divisão por ovo físico
# ---------------------------------------------------------------------------------------
def make_groups(fnames, labels):
    def ts(fn):
        p = fn.replace('.jpg', '').split('_')
        return int(p[1]) * 100000 + int(p[2][:2]) * 3600 + int(p[2][2:4]) * 60 + int(p[2][4:6])
    t = np.array([ts(f) for f in fnames])
    order = np.argsort(t)
    g = np.zeros(len(fnames), np.int64)
    cur = 0
    for i, pos in enumerate(order):
        if i and (labels[pos] != labels[order[i - 1]] or t[pos] - t[order[i - 1]] > GROUP_GAP_S):
            cur += 1
        g[pos] = cur
    return g


def make_split(groups, labels, kind, sessions, seed):
    """
    kind: 'good' | 'real' | 'pencil'.
      good   -> por sessão: 60% treino / 20% validação / 20% teste (grupos físicos)
      real   -> 100% validação/teste, balanceando nº de fotos
      pencil -> 70% treino / 30% 'pencil_eval' (relatório à parte, nunca no treino)
    """
    rng = np.random.default_rng(seed)
    split = np.array([''] * len(labels), dtype=object)
    for s in np.unique(sessions):
        gs = np.unique(groups[(kind == 'good') & (sessions == s)])
        rng.shuffle(gs)
        n = len(gs)
        n_va = max(1, int(round(0.2 * n)))
        n_te = max(1, int(round(0.2 * n)))
        for j, g in enumerate(gs):
            split[groups == g] = 'val' if j < n_va else ('test' if j < n_va + n_te else 'train')
    real_g = np.unique(groups[kind == 'real'])
    rng.shuffle(real_g)
    sizes = {g: int(np.sum(groups == g)) for g in real_g}
    cnt = {'val': 0, 'test': 0}
    ng = {'val': 0, 'test': 0}
    for g in sorted(real_g, key=lambda g: -sizes[g]):
        tgt = min(('val', 'test'), key=lambda k: (ng[k], cnt[k]))
        split[groups == g] = tgt
        cnt[tgt] += sizes[g]
        ng[tgt] += 1
    pen_g = np.unique(groups[kind == 'pencil'])
    rng.shuffle(pen_g)
    n_pe = int(round(0.3 * len(pen_g)))
    for j, g in enumerate(pen_g):
        split[groups == g] = 'pencil_eval' if j < n_pe else 'train'
    return split


# ---------------------------------------------------------------------------------------
# Pipeline de uma divisão
# ---------------------------------------------------------------------------------------
def run_split(seed, data, backbone, greedy=True):
    labels, groups, kind, sessions = data['labels'], data['groups'], data['kind'], data['sessions']
    split = make_split(groups, labels, kind, sessions, seed)
    tr = np.where(split == 'train')[0]
    tr_good = tr[labels[tr] == 0]
    rng = np.random.default_rng(seed)

    # --- Augmentation das imagens de treino (bons + lápis) ---
    aug_imgs, aug_masks, aug_morph, aug_src = [], [], [], []
    for i in tr:
        for _ in range(N_AUG):
            a_img, a_msk = augment(data['crops'][i], data['masks'][i], rng)
            masked, in28, morph = mask_and_morph(a_img, a_msk)
            aug_imgs.append(masked); aug_masks.append(in28); aug_morph.append(morph); aug_src.append(i)
    aug_src = np.array(aug_src)
    aug_patches = extract_patches(backbone, aug_imgs, aug_masks)
    aug_morph = np.array(aug_morph)

    P = data['patches']
    # Conjunto de treino expandido: originais + aumentadas
    tr_items = [(P[i], data['morph'][i], i) for i in tr] + \
               [(aug_patches[j], aug_morph[j], aug_src[j]) for j in range(len(aug_src))]
    tr_src = np.array([t[2] for t in tr_items])
    tr_y = labels[tr_src]
    tr_g = groups[tr_src]

    # --- Banco de memória final: só ovos BONS de treino (originais + aumentados) ---
    good_items = [t[0] for t in tr_items if labels[t[2]] == 0]
    bank = build_bank(good_items, seed, greedy)

    # --- Features PatchCore do treino OUT-OF-FOLD (sem o próprio ovo no banco) ---
    good_groups = np.unique(groups[tr_good])
    fold_of = {g: k % 5 for k, g in enumerate(np.random.default_rng(seed).permutation(good_groups))}
    tr_pc = np.zeros((len(tr_items), 5), np.float32)
    for f in range(5):
        bank_f = build_bank([t[0] for t, gg in zip(tr_items, tr_g)
                             if labels[t[2]] == 0 and fold_of.get(gg, -1) != f], seed + f, greedy)
        # ovos bons do fold f são pontuados contra o banco sem eles; lápis entra no fold 0
        sel = [k for k, gg in enumerate(tr_g)
               if (tr_y[k] == 0 and fold_of[gg] == f) or (tr_y[k] == 1 and f == 0)]
        if f == 0:
            pencil_sel = [k for k in sel if tr_y[k] == 1]
            good_sel = [k for k in sel if tr_y[k] == 0]
            tr_pc[good_sel] = patch_scores([tr_items[k][0] for k in good_sel], bank_f)
            tr_pc[pencil_sel] = patch_scores([tr_items[k][0] for k in pencil_sel], bank)  # lápis nunca está no banco
        else:
            tr_pc[sel] = patch_scores([tr_items[k][0] for k in sel], bank_f)
        del bank_f

    ev = np.where(np.isin(split, ['val', 'test', 'pencil_eval']))[0]
    ev_pc = patch_scores([P[i] for i in ev], bank)
    pc_all = np.full((len(labels), 5), np.nan, np.float32)
    pc_all[ev] = ev_pc

    # --- Score One-Class (só depende de ovos bons) ---
    good_mask = tr_y == 0
    s_pc = tr_pc[good_mask, 1].std() + 1e-6
    s_el = np.array([t[1][0] for t in tr_items])[good_mask].std() + 1e-6
    oc = lambda pc, mo: 0.8 * pc[:, 1] / s_pc + 0.2 * mo[:, 0] / s_el
    oc_score = np.full(len(labels), np.nan)
    oc_score[ev] = oc(pc_all[ev], data['morph'][ev])

    # --- Score Híbrido: regressão logística sobre 9 features (treino: bons + lápis aumentados) ---
    X_tr = np.hstack([tr_pc, np.array([t[1] for t in tr_items])])
    X_ev = np.hstack([pc_all[ev], data['morph'][ev]])
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000).fit(sc.transform(X_tr), tr_y)
    hyb_score = np.full(len(labels), np.nan)
    hyb_score[ev] = clf.predict_proba(sc.transform(X_ev))[:, 1]

    res = {'split': split, 'oc': oc_score, 'hyb': hyb_score, 'n_train_imgs': len(tr_items),
           'n_bank': int(bank.shape[0]), 'clf': clf, 'scaler': sc}
    for name, sco in [('oc', oc_score), ('hyb', hyb_score)]:
        for part in ['val', 'test']:
            m = split == part
            res[f'{name}_{part}_auc'] = roc_auc_score(labels[m], sco[m])
        m = (split == 'pencil_eval') | ((split == 'test') & (labels == 0))
        res[f'{name}_pencil_auc'] = roc_auc_score(labels[m], sco[m])
        # Controle de confusão por sessão: só sessão 04/Nov (bons e trincados reais da mesma sessão)
        m = np.isin(split, ['val', 'test']) & (sessions == '20261104')
        res[f'{name}_s04_auc'] = roc_auc_score(labels[m], sco[m]) if len(np.unique(labels[m])) == 2 else np.nan
    return res


def benchmark_cpu(n=50):
    """Latência de inferência em CPU (1 thread e todas as threads) da backbone + kNN."""
    bb = LightBackbone().to('cpu').eval()
    bank = F.normalize(torch.randn(CORESET_SIZE, 24 + 48), dim=1)
    x = np.random.randint(0, 255, (1, 224, 224, 3), np.uint8)
    out = {}
    for th in [1, os.cpu_count()]:
        torch.set_num_threads(th)
        for _ in range(5):
            p = bb(x).permute(0, 2, 3, 1).reshape(-1, 72)
            torch.topk(p @ bank.T, 3, 1)
        t0 = time.perf_counter()
        for _ in range(n):
            p = bb(x).permute(0, 2, 3, 1).reshape(-1, 72)
            torch.topk(p @ bank.T, 3, 1)
        out[th] = (time.perf_counter() - t0) / n * 1000
    torch.set_num_threads(os.cpu_count())
    params = sum(p.numel() for p in bb.parameters())
    return params, out


def main():
    t0 = time.time()
    set_seed(SEED)
    print('=' * 90)
    print('EXP 3 — Trincas reais na avaliação, augmentation, prevalência real e modelo leve')
    print('=' * 90)

    good = sorted(glob.glob(os.path.join('New sample', 'Good Eggs', '*.jpg')))
    bad = sorted(glob.glob(os.path.join('New sample', 'Bad eggs', '*.jpg')))
    paths = good + bad
    labels = np.array([0] * len(good) + [1] * len(bad))
    fnames = np.array([os.path.basename(p) for p in paths])
    sessions = np.array([f.split('_')[1] for f in fnames])

    # 4 threads: cada foto de 12 MP decodificada ocupa ~36 MB de RAM
    with ThreadPoolExecutor(max_workers=4) as ex:
        loc = list(ex.map(locate_egg, paths))
    keep, seen = [], set()
    for i, r in enumerate(loc):
        if r is not None and r[0] not in seen:
            seen.add(r[0]); keep.append(i)
    keep = np.array(keep)
    labels, fnames, sessions = labels[keep], fnames[keep], sessions[keep]
    crops = [loc[i][1] for i in keep]
    masks = [loc[i][2] for i in keep]
    mm = [mask_and_morph(c, m) for c, m in zip(crops, masks)]
    masked = [x[0] for x in mm]
    inner28 = [x[1] for x in mm]
    morph = np.array([x[2] for x in mm])

    kind = np.where(labels == 0, 'good', np.where(sessions == '20261108', 'pencil', 'real'))
    groups = make_groups(fnames, labels)
    print(f'\n[1] {len(labels)} fotos únicas | bons={np.sum(kind=="good")} '
          f'| trincas reais={np.sum(kind=="real")} ({len(np.unique(groups[kind=="real"]))} ovos físicos) '
          f'| trincas a lápis={np.sum(kind=="pencil")} ({len(np.unique(groups[kind=="pencil"]))} ovos físicos)')

    backbone = LightBackbone().to(DEVICE).eval()
    patches = extract_patches(backbone, masked, inner28)
    data = dict(labels=labels, groups=groups, kind=kind, sessions=sessions, crops=crops,
                masks=masks, patches=patches, morph=morph)

    # ---------------- Divisão principal (seed 42) ----------------
    r = run_split(SEED, data, backbone)
    split = r['split']
    print('\n[2] Divisão principal por ovo físico (seed 42):')
    for part in ['train', 'val', 'test', 'pencil_eval']:
        m = split == part
        print(f'   {part:12s}: {m.sum():3d} fotos | bons={np.sum(m & (labels==0)):3d} '
              f'({len(np.unique(groups[m & (labels==0)]))} ovos) | trincados={np.sum(m & (labels==1)):3d} '
              f'({len(np.unique(groups[m & (labels==1)]))} ovos; '
              f'{"lápis" if part in ("train","pencil_eval") else "reais"})')
    print(f'   Treino após augmentation: {r["n_train_imgs"]} imagens (x{N_AUG + 1}); banco de memória: {r["n_bank"]} patches')

    # Checagem explícita de vazamento: nenhum ovo físico em mais de uma partição
    leak = [g for g in np.unique(groups) if len(set(split[groups == g])) > 1]
    print(f'   Checagem de vazamento: {len(leak)} ovos físicos em mais de uma partição')
    assert not leak

    # Escolha do modelo pela VALIDAÇÃO (não pelo teste)
    best = 'hyb' if r['hyb_val_auc'] > r['oc_val_auc'] else 'oc'
    names = {'oc': 'One-Class (PatchCore leve)', 'hyb': 'Híbrido (PatchCore + Reg. Logística)'}
    print('\n[3] AUROC (IC 95% por bootstrap de ovos físicos):')
    rows_auc = []
    for k in ['oc', 'hyb']:
        for part in ['val', 'test']:
            m = split == part
            a, lo, hi = group_bootstrap_auc(labels[m], r[k][m], groups[m])
            rows_auc.append((k, part, a, lo, hi))
            print(f'   {names[k]:38s} {part:5s}: {a*100:5.1f}% [{lo*100:5.1f}%, {hi*100:5.1f}%]')
        print(f'   {names[k]:38s} só sessão 04/Nov (val+test): {r[k+"_s04_auc"]*100:5.1f}%'
              f' | trincas a lápis (relatório à parte): {r[k+"_pencil_auc"]*100:5.1f}%')
    print(f'   → Modelo escolhido pela validação: {names[best]}')

    # ---------------- Limiar por custo esperado ----------------
    print('\n[4] Limiar pelo menor custo esperado por ovo (prevalência real), avaliado no TESTE:')
    yv, sv = labels[split == 'val'], r[best][split == 'val']
    yt, st = labels[split == 'test'], r[best][split == 'test']
    gt = groups[split == 'test']
    cost_rows = []
    for prev in PREVALENCE_SCENARIOS:
        for ratio in COST_RATIOS:
            t = pick_threshold(yv, sv, prev, ratio)
            tp, fn, tn, fp = rates(yt, st, t)
            rec, rlo, rhi = wilson_ci(tp, tp + fn)
            fpr, flo, fhi = wilson_ci(fp, fp + tn)
            # decisão por ovo físico: ovo rejeitado se QUALQUER foto dele passar do limiar
            eg = pd.DataFrame({'g': gt, 'y': yt, 'p': st >= t}).groupby('g').agg(y=('y', 'max'), p=('p', 'max'))
            e_rec = eg.p[eg.y == 1].mean()
            e_fpr = eg.p[eg.y == 0].mean()
            c_model = expected_cost(1 - rec, fpr, prev, ratio)
            c_none = prev * ratio  # sem inspeção: todo ovo trincado passa
            cost_rows.append(dict(prevalencia=prev, razao_custo_FN_FP=ratio, tau=t,
                                  recall=rec, recall_ci=f'[{rlo*100:.1f}%, {rhi*100:.1f}%]',
                                  fpr=fpr, fpr_ci=f'[{flo*100:.1f}%, {fhi*100:.1f}%]',
                                  recall_por_ovo=e_rec, fpr_por_ovo=e_fpr,
                                  custo_por_1000_ovos_modelo=1000 * c_model,
                                  custo_por_1000_ovos_sem_inspecao=1000 * c_none,
                                  economia_pct=100 * (1 - c_model / c_none)))
            if prev == PREVALENCE_BASE:
                print(f'   π={prev:.0%} C_FN:C_FP={ratio:>2}:1 | τ={t:7.3f} | recall={rec*100:5.1f}% '
                      f'[{rlo*100:4.1f}–{rhi*100:5.1f}] | FPR={fpr*100:5.1f}% [{flo*100:4.1f}–{fhi*100:4.1f}] '
                      f'| custo/1000 ovos: modelo={1000*c_model:6.1f} vs sem inspeção={1000*c_none:6.1f}')
    cost_df = pd.DataFrame(cost_rows)

    # ---------------- Leveza / CPU ----------------
    params, lat = benchmark_cpu()
    print(f'\n[5] Modelo leve: backbone com {params/1e3:.0f} mil parâmetros (MobileNetV3-Small truncada no bloco 8); '
          f'banco = {CORESET_SIZE}x72 float16 ≈ {CORESET_SIZE*72*2/1e6:.1f} MB')
    for th, ms in lat.items():
        print(f'   Latência CPU ({th} thread{"s" if th > 1 else ""}): {ms:.1f} ms/imagem (backbone + kNN)')

    # ---------------- Salvar ----------------
    pd.DataFrame({
        'filename': fnames, 'session_date': sessions, 'egg_group_id': groups, 'true_label': labels,
        'crack_type': np.where(kind == 'good', 'none', kind), 'split': split,
        'exp3_oneclass_score': np.round(r['oc'], 6), 'exp3_hybrid_prob': np.round(r['hyb'], 6),
    }).to_csv('exp3_results.csv', index=False)
    cost_df.round(4).to_csv('exp3_cost_thresholds.csv', index=False)
    summary = {
        'chosen_model': best,
        'auroc': [dict(model=k, part=p, auc=a, ci_low=lo, ci_high=hi) for k, p, a, lo, hi in rows_auc],
        'session04_auc': {k: r[k + '_s04_auc'] for k in ['oc', 'hyb']},
        'pencil_auc': {k: r[k + '_pencil_auc'] for k in ['oc', 'hyb']},
        'cpu_latency_ms': lat, 'backbone_params': params,
    }
    with open('exp3_summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f'\n✅ Divisão principal salva: exp3_results.csv, exp3_cost_thresholds.csv, exp3_summary.json '
          f'({time.time()-t0:.0f}s)')

    # ---------------- Repetições (depois de salvar a divisão principal) ----------------
    print(f'\n[6] Robustez: {N_REPEATS} divisões aleatórias diferentes por ovo físico '
          f'(banco por amostragem aleatória para acelerar)...')
    reps = []
    for s in range(N_REPEATS):
        t_rep = time.time()
        rr = run_split(1000 + s, data, backbone, greedy=False)
        reps.append({'seed': 1000 + s, **{k: rr[k] for k in rr if k.endswith('_auc')}})
        pd.DataFrame(reps).round(4).to_csv('exp3_repeated_splits.csv', index=False)  # salva a cada repetição
        print(f'   rep {s+1:2d}/{N_REPEATS}: One-Class teste={rr["oc_test_auc"]*100:5.1f}% | '
              f'Híbrido teste={rr["hyb_test_auc"]*100:5.1f}% ({time.time()-t_rep:.0f}s)')
    reps = pd.DataFrame(reps)
    summary['repeated'] = {}
    for k in ['oc', 'hyb']:
        summary['repeated'][k] = {}
        for col in ['test_auc', 's04_auc', 'pencil_auc']:
            v = reps[f'{k}_{col}'].dropna()
            summary['repeated'][k][col] = dict(median=float(v.median()), p10=float(v.quantile(.1)),
                                               p90=float(v.quantile(.9)), n=int(len(v)))
            print(f'   {names[k]:38s} {col:10s}: mediana {v.median()*100:5.1f}% '
                  f'(P10–P90: {v.quantile(.1)*100:5.1f}%–{v.quantile(.9)*100:5.1f}%)')
    with open('exp3_summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f'\n✅ Concluído: + exp3_repeated_splits.csv ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
