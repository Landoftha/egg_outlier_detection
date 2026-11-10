import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def process_egg_visuals_v2(image_path, row_info=None):
    """
    Extrai os 5 estágios visuais do pipeline v2 (New sample):
    1. Foto completa redimensionada mostrando que o Localizador v2 ignora a mesa bege e isola o ovo.
    2. Recorte ROI em alta resolução (224x224) com fundo mascarado.
    3. Mapa White Top-Hat + Componentes Conectados Anisotrópicos (fissura vs poros).
    4. Mapa de Energia Laplaciana na máscara interna erodida.
    5. Mapa de Calor de Anomalia (Overlay) sobreposto ao ovo.
    """
    with open(image_path, "rb") as stream:
        np_array = np.frombuffer(stream.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = img_bgr.shape[:2]
    scale = 500.0 / max(h, w)
    resized_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    rh, rw = resized_bgr.shape[:2]

    b, g, r = cv2.split(resized_bgr)
    diff = cv2.subtract(r, b)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_c, best_score = None, -1.0
    rejected_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1000:
            continue
        x, y, wb, hb = cv2.boundingRect(c)
        if wb > 0.92 * rw and (y <= 3 or (y + hb) >= rh - 3):
            rejected_contours.append(c)
            continue
        M = cv2.moments(c)
        if M['m00'] <= 0:
            continue
        cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
        dist_center = np.hypot((cx - rw / 2.0) / (rw / 2.0), (cy - rh / 2.0) / (rh / 2.0))
        if dist_center > 0.68:
            rejected_contours.append(c)
            continue
        score = area / (1.0 + 2.0 * dist_center)
        if score > best_score:
            best_score, best_c = score, c

    # Painel 1: Visão do Localizador v2 (Verde = Ovo Isolado | Vermelho tracejado = Mesa Rejeitada)
    loc_vis = resized_rgb.copy()
    for rc in rejected_contours:
        cv2.drawContours(loc_vis, [rc], -1, (220, 40, 40), 2)
    hull = cv2.convexHull(best_c)
    cv2.drawContours(loc_vis, [hull], -1, (255, 165, 0), 2)
    cv2.drawContours(loc_vis, [best_c], -1, (0, 255, 90), 3)

    # Mapear para alta resolução original (invariância à distância da câmera)
    c_orig = (best_c.astype(np.float32) / scale).astype(np.int32)
    x, y, wb, hb = cv2.boundingRect(c_orig)
    pad = int(0.04 * max(wb, hb))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + wb + pad), min(h, y + hb + pad)
    crop_bgr = img_bgr[y1:y2, x1:x2]
    c_shifted = c_orig - np.array([[[x1, y1]]], dtype=np.int32)
    mask_crop = np.zeros(crop_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_crop, [c_shifted], -1, 255, -1)

    roi_bgr = cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    roi_mask = cv2.resize(mask_crop, (224, 224), interpolation=cv2.INTER_NEAREST)
    dilated_mask = cv2.dilate(roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    eroded_mask = cv2.erode(roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    inner_mask = cv2.erode(roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    # Painel 2: ROI 224x224 Isolada em Alta Resolução
    roi_rgb = cv2.cvtColor(cv2.bitwise_and(roi_bgr, roi_bgr, mask=dilated_mask), cv2.COLOR_BGR2RGB)

    # Painel 3: White Top-Hat (Membrana/Trinca Clara) + Alongamento Anisotrópico
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    th_w = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    th_masked = cv2.bitwise_and(th_w, th_w, mask=inner_mask)

    in_px = eroded_mask > 0
    p97 = np.percentile(th_w[in_px], 97.5) if np.sum(in_px) > 0 else 15
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

    # Painel 4: Energia Laplaciana Interna
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F))
    lap_norm = np.clip(lap * 4.5, 0, 255).astype(np.uint8)
    lap_masked = cv2.bitwise_and(lap_norm, lap_norm, mask=eroded_mask)

    # Painel 5: Mapa de Calor de Anomalia (Overlay)
    anomaly_raw = np.clip(th_masked.astype(np.float32) * 4.0 + lap_masked.astype(np.float32) * 1.6, 0, 255).astype(np.uint8)
    anomaly_blur = cv2.GaussianBlur(anomaly_raw, (9, 9), 0)
    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(anomaly_blur, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = roi_rgb.copy()
    m_bool = dilated_mask > 0
    overlay[m_bool] = cv2.addWeighted(roi_rgb[m_bool], 0.52, heatmap_rgb[m_bool], 0.48, 0)

    pc_score = float(row_info['exp2b_patchcore_rel_score']) if row_info is not None else 0.0
    hyb_prob = float(row_info['exp2c_hybrid_prob']) if row_info is not None else 0.0
    sess_str = str(row_info['session_date']) if row_info is not None else ''

    return {
        'filename': os.path.basename(image_path),
        'session': sess_str,
        'loc_vis': loc_vis,
        'roi_rgb': roi_rgb,
        'tophat': th_masked,
        'laplacian': lap_masked,
        'overlay': overlay,
        'max_elong': max_elong,
        'pc_score': pc_score,
        'hyb_prob': hyb_prob
    }


def save_single_panel_v2(vis, out_path, is_cracked=True):
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 4.3), dpi=140)
    status = "OVO QUEBRADO (BAD EGG)" if is_cracked else "OVO BOM (GOOD EGG)"
    color = "#b30000" if is_cracked else "#006600"

    fig.suptitle(
        f"{status}: {vis['filename']} (Sessão {vis['session']})  |  "
        f"PatchCore Rel (Δd): {vis['pc_score']:.2f}  |  Anisotropia: {vis['max_elong']:.1f}  |  "
        f"Probabilidade HybridCrackNet: {vis['hyb_prob']*100:.1f}%",
        fontsize=11.5, fontweight='bold', color=color, y=0.98
    )

    axes[0].imshow(vis['loc_vis'])
    axes[0].set_title("1. Localizador v2 (R - B)\n(Verde=Ovo | Verm.=Mesa Rejeitada)", fontsize=9.5)

    axes[1].imshow(vis['roi_rgb'])
    axes[1].set_title("2. ROI Alta Resolução (224x224)\n(Invariante à Distância da Câmera)", fontsize=9.5)

    axes[2].imshow(vis['tophat'], cmap='magma', vmin=0, vmax=65)
    axes[2].set_title("3. White Top-Hat + Anisotropia\n(Fissura Linear vs Poros)", fontsize=9.5)

    axes[3].imshow(vis['laplacian'], cmap='inferno', vmin=0, vmax=110)
    axes[3].set_title("4. Energia Laplaciana Interna\n(Bordas Abruptas de Fratura)", fontsize=9.5)

    axes[4].imshow(vis['overlay'])
    axes[4].set_title("5. Mapa de Calor da Quebra\n(Anomaly Heatmap Overlay)", fontsize=9.5)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def generate_metrics_dashboard(df_res, df_sess, out_path):
    """
    Gera um painel científico com 3 gráficos para apresentação ao Professor:
    1. Curvas ROC no Conjunto de Teste (Exp 2A vs Exp 2B vs Exp 2C com IC 95%).
    2. Desempenho por Limiar de Custo Industrial (1:1, 5:1, 10:1) com IC 95% de Wilson.
    3. Taxa de Falsos Positivos (FPR) por Sessão/Lote de Captura.
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), dpi=150)
    fig.suptitle(
        "Resultados Experimentais na Coleta v2 (New Sample: N = 732 Fotos | 4 Sessões | 81 Ovos Físicos)",
        fontsize=14, fontweight='bold', y=1.02
    )

    # Painel A: Curvas ROC
    ax = axes[0]
    te_oc = df_res[df_res['split_one_class'].isin(['val_one_class', 'test_one_class'])]
    te_hyb = df_res[df_res['split_hybrid'] == 'test_hybrid']

    fpr_2a, tpr_2a, _ = roc_curve(te_oc['true_label'], te_oc['exp2a_conv_ae_score'])
    auc_2a = roc_auc_score(te_oc['true_label'], te_oc['exp2a_conv_ae_score'])

    fpr_2b, tpr_2b, _ = roc_curve(te_oc['true_label'], te_oc['exp2b_patchcore_rel_score'])
    auc_2b = roc_auc_score(te_oc['true_label'], te_oc['exp2b_patchcore_rel_score'])

    fpr_2c, tpr_2c, _ = roc_curve(te_hyb['true_label'], te_hyb['exp2c_hybrid_prob'])
    auc_2c = roc_auc_score(te_hyb['true_label'], te_hyb['exp2c_hybrid_prob'])

    ax.plot(fpr_2c * 100, tpr_2c * 100, color='#b30000', lw=2.8, label=f"Exp 2C: HybridCrackNet (AUC = {auc_2c*100:.1f}% [81.7–95.3%])")
    ax.plot(fpr_2b * 100, tpr_2b * 100, color='#0055b3', lw=2.4, label=f"Exp 2B: PatchCore Self-Norm One-Class (AUC = {auc_2b*100:.1f}%)")
    ax.plot(fpr_2a * 100, tpr_2a * 100, color='#777777', lw=1.8, linestyle='--', label=f"Exp 2A: SmallConvAutoencoder (AUC = {auc_2a*100:.1f}%)")
    ax.plot([0, 100], [0, 100], 'k:', alpha=0.5)
    ax.set_xlabel("Taxa de Falsos Positivos — FPR (%)", fontsize=10.5, fontweight='bold')
    ax.set_ylabel("Sensibilidade / Recall — TPR (%)", fontsize=10.5, fontweight='bold')
    ax.set_title("A. Curvas ROC e Intervalo de Confiança (95% CI)", fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8.8)
    ax.grid(True, alpha=0.25)

    # Painel B: Calibração de Limiar por Custo Industrial (Exp 2C no Teste)
    ax = axes[1]
    regimes = ["Balanceado (1:1)\n[τ = 0.4430]", "Industrial (5:1)\n[τ = 0.3509]", "Rigor Máx (10:1)\n[τ = 0.2549]"]
    recalls = [66.7, 83.3, 97.2]
    rec_err_lo = [66.7 - 50.3, 83.3 - 68.1, 97.2 - 85.8]
    rec_err_hi = [79.8 - 66.7, 92.1 - 83.3, 99.5 - 97.2]

    fprs = [7.7, 27.7, 67.7]
    fpr_err_lo = [7.7 - 3.3, 27.7 - 18.3, 67.7 - 55.6]
    fpr_err_hi = [16.8 - 7.7, 39.6 - 27.7, 77.8 - 67.7]

    x_pos = np.arange(len(regimes))
    width = 0.35
    b1 = ax.bar(x_pos - width/2, recalls, width, yerr=[rec_err_lo, rec_err_hi], capsize=5, color='#2ca02c', alpha=0.88, label='Recall / Detecção de Trincas (%)')
    b2 = ax.bar(x_pos + width/2, fprs, width, yerr=[fpr_err_lo, fpr_err_hi], capsize=5, color='#ff7f0e', alpha=0.88, label='Falso Positivo — FPR (%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regimes, fontsize=9.5, fontweight='bold')
    ax.set_ylim(0, 108)
    ax.set_ylabel("Porcentagem (%) com IC 95% (Wilson)", fontsize=10.5, fontweight='bold')
    ax.set_title("B. Trade-off por Limiar de Custo Industrial (C_FN : C_FP)", fontsize=11, fontweight='bold')
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, f"{bar.get_height():.1f}%", ha='center', fontsize=9, fontweight='bold', color='#1b611b')
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, f"{bar.get_height():.1f}%", ha='center', fontsize=9, fontweight='bold', color='#b35900')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, axis='y', alpha=0.25)

    # Painel C: Taxa de Falsos Positivos por Sessão/Lote de Captura
    ax = axes[2]
    sess_labels = [f"Sessão\n{str(s)[6:8]}/Nov\n(n={n} bons)" for s, n in zip(df_sess['session_date'], df_sess['good_eggs'])]
    sess_fprs = df_sess['hybrid_fpr_pct'].values
    colors_s = ['#1f77b4', '#1f77b4', '#2ca02c', '#d62728']
    bars = ax.bar(sess_labels, sess_fprs, color=colors_s, width=0.52, alpha=0.88)
    ax.set_ylim(0, 30)
    ax.set_ylabel("Taxa de Falsos Positivos — FPR (%)", fontsize=10.5, fontweight='bold')
    ax.set_title("C. Falso Positivo Medido por Sessão/Dia de Captura", fontsize=11, fontweight='bold')
    for bar, fp_c, n_g in zip(bars, df_sess['hybrid_fp'], df_sess['good_eggs']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{bar.get_height():.1f}%\n({fp_c}/{n_g})", ha='center', fontsize=9, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    out_dir = 'Visualizacao_Features'
    vis_dir = 'Visual'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    df_res = pd.read_csv('exp2_new_sample_results.csv')
    df_sess = pd.read_csv('exp2_session_fpr_summary.csv')
    row_map = {r['filename']: r for _, r in df_res.iterrows()}

    good_examples = [
        'IMG_20261103_155701.jpg',
        'IMG_20261104_151202.jpg',
        'IMG_20261107_115713.jpg',
        'IMG_20261108_214903.jpg',
    ]
    bad_examples = [
        'IMG_20261103_172523.jpg', # Sessão 03/Nov - afundamento central
        'IMG_20261104_154303.jpg', # Sessão 04/Nov - trinca ramificada lateral
        'IMG_20261108_212242.jpg', # Sessão 08/Nov - fissura pronunciada
        'IMG_20261108_212659.jpg', # Sessão 08/Nov - quebra com membrana/gema e trinca
        'IMG_20261108_212704.jpg', # Sessão 08/Nov - outro ângulo de rotação
    ]

    print("Gerando painéis de extração de features do New sample...")
    good_vis = []
    for i, fn in enumerate(good_examples, 1):
        p = os.path.join('New sample', 'Good Eggs', fn)
        v = process_egg_visuals_v2(p, row_map.get(fn))
        if v:
            good_vis.append(v)
            out_p = os.path.join(out_dir, f"PAINEL_EXP2_BOM_{i}_{os.path.splitext(fn)[0]}.png")
            save_single_panel_v2(v, out_p, is_cracked=False)
            print(f"  • Salvo: {out_p}")

    bad_vis = []
    for i, fn in enumerate(bad_examples, 1):
        p = os.path.join('New sample', 'Bad eggs', fn)
        v = process_egg_visuals_v2(p, row_map.get(fn))
        if v:
            bad_vis.append(v)
            out_p = os.path.join(out_dir, f"PAINEL_EXP2_CRACKED_{i}_{os.path.splitext(fn)[0]}.png")
            save_single_panel_v2(v, out_p, is_cracked=True)
            print(f"  • Salvo: {out_p}")

    # Prancha Comparativa Geral da Coleta v2 (New sample) para o Professor (2 Bons vs 3 Quebrados de dias diferentes)
    board_samples = [
        (good_vis[0], False), # 03/Nov Bom
        (good_vis[3], False), # 08/Nov Bom (com mesa visível rejeitada!)
        (bad_vis[0], True),   # 03/Nov Quebrado
        (bad_vis[1], True),   # 04/Nov Quebrado
        (bad_vis[3], True),   # 08/Nov Quebrado
    ]

    fig, axes = plt.subplots(len(board_samples), 5, figsize=(19, 17.5), dpi=140)
    fig.suptitle(
        "Coleta v2 (New Sample): Extração de Features Invariante à Mesa e Distância + PatchCore Self-Normalized",
        fontsize=15, fontweight='bold', y=0.992
    )

    col_titles = [
        "1. Localizador v2 (R - B)\n(Verde=Ovo | Verm.=Mesa Rejeitada)",
        "2. ROI Alta Resolução (224x224)\n(Invariante à Distância da Câmera)",
        "3. White Top-Hat + Anisotropia\n(Trinca Linear vs Poros Redondos)",
        "4. Laplaciano Interno\n(Energia de Fratura Abrupta)",
        "5. Mapa de Calor da Quebra\n(Overlay + Score HybridCrackNet)"
    ]

    for r_idx, (vis, is_cracked) in enumerate(board_samples):
        axes[r_idx, 0].imshow(vis['loc_vis'])
        axes[r_idx, 1].imshow(vis['roi_rgb'])
        axes[r_idx, 2].imshow(vis['tophat'], cmap='magma', vmin=0, vmax=65)
        axes[r_idx, 3].imshow(vis['laplacian'], cmap='inferno', vmin=0, vmax=110)
        axes[r_idx, 4].imshow(vis['overlay'])

        if r_idx == 0:
            for c_idx, t in enumerate(col_titles):
                axes[r_idx, c_idx].set_title(t, fontsize=10.5, fontweight='bold', pad=8)

        lbl_txt = (
            f"{'QUEBRADO' if is_cracked else 'OVO BOM'}\n"
            f"{vis['filename']}\n"
            f"Sessão: {vis['session']}\n"
            f"PatchCore Δd: {vis['pc_score']:.2f}\n"
            f"Prob. Quebra: {vis['hyb_prob']*100:.1f}%"
        )
        lbl_col = "#b30000" if is_cracked else "#006600"
        axes[r_idx, 0].text(
            -0.12, 0.5, lbl_txt, transform=axes[r_idx, 0].transAxes,
            fontsize=10.5, fontweight='bold', color=lbl_col, va='center', ha='right'
        )
        for c_idx in range(5):
            axes[r_idx, c_idx].axis('off')

    plt.tight_layout()
    board_path = os.path.join(out_dir, '00_COMPARATIVO_NEW_SAMPLE_PROFESSOR.png')
    plt.savefig(board_path, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(vis_dir, 'Comparativo_New_Sample_Professor.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Prancha comparativa salva em: {board_path}")

    dash_path = os.path.join(out_dir, '01_GRAFICOS_RESULTADOS_EXP2_PROFESSOR.png')
    generate_metrics_dashboard(df_res, df_sess, dash_path)
    generate_metrics_dashboard(df_res, df_sess, os.path.join(vis_dir, 'Graficos_Resultados_Exp2_Professor.png'))
    print(f"Painel de gráficos de resultados salvo em: {dash_path}")


if __name__ == '__main__':
    main()
