import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_egg_visuals(image_path):
    """
    Executa todas as etapas de localização e extração de features de rachadura,
    retornando as imagens intermediárias para visualização didática.
    """
    with open(image_path, "rb") as stream:
        bytes_array = bytearray(stream.read())
    np_array = np.asarray(bytes_array, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    scale = 600.0 / max(h, w)
    resized_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    # 1. Separação Espectral (R - B) + Otsu
    b_chan, g_chan, r_chan = cv2.split(resized_bgr)
    color_diff = cv2.subtract(r_chan, b_chan)
    _, thresh = cv2.threshold(color_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_c = max(contours, key=cv2.contourArea)
    egg_mask = np.zeros_like(mask)
    cv2.drawContours(egg_mask, [best_c], -1, 255, -1)

    area = float(cv2.contourArea(best_c))
    hull = cv2.convexHull(best_c)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 1.0

    # Dilatar e erodir máscara
    dilated_mask = cv2.dilate(egg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    eroded_mask = cv2.erode(egg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    # Recorte (Bounding Box) para focar todas as visualizações no ovo
    x, y, wb, hb = cv2.boundingRect(dilated_mask)
    pad = 15
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(resized_bgr.shape[1], x + wb + pad), min(resized_bgr.shape[0], y + hb + pad)

    # Painel 1: Contorno Real (Verde) vs Convex Hull (Vermelho) mostrando perda de Solidez
    contour_vis = resized_rgb.copy()
    cv2.drawContours(contour_vis, [hull], -1, (255, 50, 50), 2)   # Vermelho = Casca Convexa Ideal
    cv2.drawContours(contour_vis, [best_c], -1, (0, 255, 100), 2) # Verde = Contorno Real do Ovo
    contour_crop = contour_vis[y1:y2, x1:x2]

    # Painel 2: ROI Isolada (Fundo removido via Máscara Espectral R-B)
    masked_rgb = cv2.bitwise_and(resized_rgb, resized_rgb, mask=dilated_mask)
    roi_crop = masked_rgb[y1:y2, x1:x2]

    # Painel 3: Transformada White Top-Hat (Membrana Branca / Fissura Clara)
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    inner_mask = cv2.erode(egg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    tophat_masked = cv2.bitwise_and(tophat, tophat, mask=inner_mask)
    white_membrane_ratio = float(np.sum((tophat > 18) & (inner_mask > 0)) / area) if area > 0 else 0.0
    tophat_crop = tophat_masked[y1:y2, x1:x2]

    # Painel 4: Energia Laplaciana (Bordas Abruptas de Fratura Interna)
    laplacian = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F))
    laplacian_norm = np.clip(laplacian * 4.0, 0, 255).astype(np.uint8)
    laplacian_masked = cv2.bitwise_and(laplacian_norm, laplacian_norm, mask=eroded_mask)
    laplacian_energy = float(np.mean(laplacian[eroded_mask > 0])) if np.sum(eroded_mask > 0) > 0 else 0.0
    laplacian_crop = laplacian_masked[y1:y2, x1:x2]

    # Painel 5: Mapa de Calor Combinado (Anomaly Heatmap Overlay) sobreposto ao ovo
    anomaly_map = np.clip(tophat_masked.astype(np.float32) * 3.5 + laplacian_masked.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    anomaly_map = cv2.GaussianBlur(anomaly_map, (9, 9), 0)
    heatmap_bgr = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    # Aplicar heatmap apenas dentro da máscara do ovo
    overlay = roi_crop.copy()
    mask_crop = dilated_mask[y1:y2, x1:x2] > 0
    heatmap_crop = heatmap_rgb[y1:y2, x1:x2]
    overlay[mask_crop] = cv2.addWeighted(roi_crop[mask_crop], 0.55, heatmap_crop[mask_crop], 0.45, 0)

    return {
        'filename': os.path.basename(image_path),
        'contour_crop': contour_crop,
        'roi_crop': roi_crop,
        'tophat_crop': tophat_crop,
        'laplacian_crop': laplacian_crop,
        'overlay_crop': overlay,
        'solidity': solidity,
        'solidity_loss': 1.0 - solidity,
        'white_membrane_ratio': white_membrane_ratio,
        'laplacian_energy': laplacian_energy
    }


def save_single_egg_panel(vis, out_path, is_cracked=True):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), dpi=140)
    status_str = "OVO QUEBRADO (CRACKED)" if is_cracked else "OVO BOM (NORMAL)"
    title_color = "#b30000" if is_cracked else "#006600"

    fig.suptitle(
        f"{status_str}: {vis['filename']}  |  Solidez: {vis['solidity']:.3f} (Perda: {vis['solidity_loss']:.4f})  |  "
        f"Top-Hat (Membrana Exposta): {vis['white_membrane_ratio']:.4f}  |  Energia Laplaciana: {vis['laplacian_energy']:.2f}",
        fontsize=12, fontweight='bold', color=title_color, y=0.98
    )

    axes[0].imshow(vis['contour_crop'])
    axes[0].set_title("1. Localizador + Convex Hull\n(Verde=Contorno | Vermelho=Hull)", fontsize=9.5)

    axes[1].imshow(vis['roi_crop'])
    axes[1].set_title("2. Recorte ROI Isolado\n(Filtro Espectral R - B)", fontsize=9.5)

    axes[2].imshow(vis['tophat_crop'], cmap='magma', vmin=0, vmax=80)
    axes[2].set_title("3. Feature: White Top-Hat\n(Membrana Clara / Fissura)", fontsize=9.5)

    axes[3].imshow(vis['laplacian_crop'], cmap='inferno', vmin=0, vmax=120)
    axes[3].set_title("4. Feature: Laplaciano\n(Bordas Internas Abruptas)", fontsize=9.5)

    axes[4].imshow(vis['overlay_crop'])
    axes[4].set_title("5. Mapa de Calor de Anomalia\n(Região da Quebra Destacada)", fontsize=9.5)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    dataset_dir = 'Dataset'
    out_dir = 'Visualizacao_Features'
    os.makedirs(out_dir, exist_ok=True)

    cracked_files = sorted(
        [f for f in os.listdir(dataset_dir) if f.lower().startswith('cracked')],
        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
    )
    good_files = sorted([f for f in os.listdir(dataset_dir) if not f.lower().startswith('cracked')])[:4]

    print(f"Gerando painéis visuais de extração de features em '{out_dir}/'...")

    cracked_vis_list = []
    for f in cracked_files:
        vis = process_egg_visuals(os.path.join(dataset_dir, f))
        if vis:
            cracked_vis_list.append(vis)
            out_name = f"PAINEL_CRACKED_{os.path.splitext(f)[0].replace(' ', '_')}.png"
            save_single_egg_panel(vis, os.path.join(out_dir, out_name), is_cracked=True)
            print(f"  • Salvo: {out_name}")

    good_vis_list = []
    for i, f in enumerate(good_files, 1):
        vis = process_egg_visuals(os.path.join(dataset_dir, f))
        if vis:
            good_vis_list.append(vis)
            out_name = f"PAINEL_BOM_EXEMPLO_{i}_{os.path.splitext(f)[0]}.png"
            save_single_egg_panel(vis, os.path.join(out_dir, out_name), is_cracked=False)
            print(f"  • Salvo: {out_name}")

    # Criar Prancha Comparativa Geral para Apresentação ao Professor (2 Ovos Bons vs 3 Ovos Quebrados)
    samples = [
        (good_vis_list[1], False),
        (good_vis_list[2], False),
        (cracked_vis_list[0], True), # cracked 1
        (cracked_vis_list[4], True), # cracked 5
        (cracked_vis_list[6], True), # cracked 7
    ]

    fig, axes = plt.subplots(len(samples), 5, figsize=(18, 3.3 * len(samples)), dpi=150)
    fig.suptitle(
        "Comparativo de Extração de Features: Ovos Bons vs. Ovos Quebrados (Pipeline de Visão Computacional)",
        fontsize=15, fontweight='bold', y=0.995
    )

    col_titles = [
        "1. Contorno (Verde) vs Hull (Verm.)",
        "2. Recorte ROI (Fundo Removido)",
        "3. White Top-Hat (Membrana/Fissura)",
        "4. Laplaciano (Fratura Linear)",
        "5. Mapa de Calor da Quebra (Overlay)"
    ]

    for row_idx, (vis, is_crk) in enumerate(samples):
        tag = "QUEBRADO" if is_crk else "BOM (NORMAL)"
        row_label = f"{tag}\n{vis['filename']}\nSolidez: {vis['solidity']:.3f}\nTopHat: {vis['white_membrane_ratio']:.4f}"

        axes[row_idx, 0].imshow(vis['contour_crop'])
        axes[row_idx, 1].imshow(vis['roi_crop'])
        axes[row_idx, 2].imshow(vis['tophat_crop'], cmap='magma', vmin=0, vmax=80)
        axes[row_idx, 3].imshow(vis['laplacian_crop'], cmap='inferno', vmin=0, vmax=120)
        axes[row_idx, 4].imshow(vis['overlay_crop'])

        for col_idx in range(5):
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=10.5, fontweight='bold')
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

        axes[row_idx, 0].set_ylabel(
            row_label, fontsize=9.5, fontweight='bold',
            color=('#b30000' if is_crk else '#006600'), rotation=0, labelpad=65, va='center'
        )

    plt.tight_layout()
    summary_path = os.path.join(out_dir, "00_COMPARATIVO_GERAL_PROFESSOR.png")
    plt.savefig(summary_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nPrancha comparativa principal salva em: '{summary_path}'")


if __name__ == '__main__':
    main()
