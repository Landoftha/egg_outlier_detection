import os
import glob
import cv2
import numpy as np
import pandas as pd

def segment_egg(img, target_size=800):
    """
    Segmenta o ovo da imagem de forma robusta e invariante à cor do fundo (mousepad azul)
    e toalha/mesa nos cantos, utilizando a forte separação espectral (Canal Vermelho - Canal Azul)
    e seleção por proximidade geométrica e solidez.
    """
    h, w = img.shape[:2]
    scale = float(target_size) / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Ovos são marrons/alaranjados (R alto, B baixo). O mousepad é azul (B alto, R baixo).
    # A diferença (R - B) isola o ovo perfeitamente tanto da base azul quanto da toalha.
    b_chan, _, r_chan = cv2.split(resized)
    color_diff = cv2.subtract(r_chan, b_chan)

    # Limiarização de Otsu sobre o mapa de diferença cromática
    _, thresh = cv2.threshold(color_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Operações morfológicas para preencher brilhos especulares e eliminar pequenos ruídos
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, resized

    total_area = thresh.shape[0] * thresh.shape[1]
    center_y, center_x = thresh.shape[0] / 2.0, thresh.shape[1] / 2.0

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        # O ovo normalmente ocupa entre 1% e 50% da área da imagem
        if area < 0.01 * total_area or area > 0.55 * total_area:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        M = cv2.moments(c)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            dist_to_center = np.hypot(cx - center_x, cy - center_y)
            candidates.append((c, area, solidity, dist_to_center))

    if not candidates:
        # Fallback: pega o maior contorno disponível
        best_contour = max(contours, key=cv2.contourArea)
    else:
        # Prioriza contornos convexos (>0.85 de solidez), mais próximos do centro e de boa área
        candidates.sort(key=lambda item: (item[2] > 0.85, -item[3], item[1]), reverse=True)
        best_contour = candidates[0][0]

    # Cria máscara binária sólida do ovo
    egg_mask = np.zeros_like(thresh)
    cv2.drawContours(egg_mask, [best_contour], -1, 255, -1)

    return best_contour, egg_mask, resized

def extract_features_from_egg(image_path):
    """
    Extrai características geométricas, cromáticas e de textura/defeitos
    de uma única imagem de ovo.
    """
    # Carregamento seguro para Windows com caminhos acentuados
    with open(image_path, "rb") as stream:
        bytes_array = bytearray(stream.read())
    np_array = np.asarray(bytes_array, dtype=np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img is None:
        return None

    contour, mask, resized = segment_egg(img, target_size=800)
    if contour is None or mask is None:
        return None

    features = {'filename': os.path.basename(image_path)}

    # 1. CARACTERÍSTICAS GEOMÉTRICAS (Invariantes à rotação do celular)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    
    # Solidez (razão entre a área e a casca convexa)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Circularidade (4*pi*Area / Perímetro^2)
    circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    # Aspect Ratio e Excentricidade via Ajuste de Elipse
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        _, (axis1, axis2), _ = ellipse
        major_axis = max(axis1, axis2)
        minor_axis = min(axis1, axis2)
        aspect_ratio = float(major_axis / minor_axis) if minor_axis > 0 else 1.0
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor_axis / major_axis) ** 2))) if major_axis > 0 else 0.0
    else:
        # Fallback para MinAreaRect
        rect = cv2.minAreaRect(contour)
        w_r, h_r = rect[1]
        major_axis = max(w_r, h_r)
        minor_axis = min(w_r, h_r)
        aspect_ratio = float(major_axis / minor_axis) if minor_axis > 0 else 1.0
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor_axis / major_axis) ** 2))) if major_axis > 0 else 0.0

    features.update({
        'area': area,
        'perimeter': perimeter,
        'solidity': solidity,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'eccentricity': eccentricity
    })

    # 2. CARACTERÍSTICAS CROMÁTICAS (Apenas na casca do ovo)
    mean_bgr, std_bgr = cv2.meanStdDev(resized, mask=mask)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    mean_hsv, std_hsv = cv2.meanStdDev(hsv, mask=mask)

    features.update({
        'mean_blue': float(mean_bgr[0][0]),
        'mean_green': float(mean_bgr[1][0]),
        'mean_red': float(mean_bgr[2][0]),
        'std_red': float(std_bgr[2][0]),
        'mean_hue': float(mean_hsv[0][0]),
        'mean_sat': float(mean_hsv[1][0]),
        'mean_val': float(mean_hsv[2][0]),
        'std_val': float(std_hsv[2][0])
    })

    # 3. CARACTERÍSTICAS DE TEXTURA E DEFEITOS (Trincas, Rachaduras e Manchas)
    # Erode a máscara para excluir o contorno externo da casca e sombras de borda
    eroded_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    eroded_area = float(np.sum(eroded_mask > 0))

    if eroded_area > 0:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Densidade de bordas internas (Canny aplicado no interior do ovo)
        edges = cv2.Canny(gray, 40, 100)
        internal_edges = float(np.sum((edges > 0) & (eroded_mask > 0)) / eroded_area)

        # Detecção de manchas escuras/sardas/sujeira compensando a iluminação curva 3D
        bg_smooth = cv2.medianBlur(gray, 31)
        diff_spots = cv2.subtract(bg_smooth, gray)
        # Manchas nítidas têm contraste significativo em relação à casca adjacente
        spot_pixels = float(np.sum((diff_spots > 20) & (eroded_mask > 0)))
        spot_density = spot_pixels / eroded_area

        # Desvio padrão residual de textura (mede rugosidade/desuniformidade da casca)
        diff_residual = cv2.absdiff(gray, bg_smooth)
        mean_res, std_res = cv2.meanStdDev(diff_residual, mask=eroded_mask)
        texture_roughness = float(std_res[0][0])
    else:
        internal_edges = 0.0
        spot_density = 0.0
        texture_roughness = 0.0

    features.update({
        'internal_edge_density': internal_edges,
        'spot_density': spot_density,
        'texture_roughness': texture_roughness
    })

    return features

def main():
    dataset_dir = os.path.join(os.getcwd(), 'Dataset')
    image_paths = sorted(glob.glob(os.path.join(dataset_dir, '*.jpg')))

    print(f"Encontradas {len(image_paths)} imagens no Dataset.")
    print("Iniciando extração robusta de características...")

    all_features = []
    for i, path in enumerate(image_paths):
        feats = extract_features_from_egg(path)
        if feats is not None:
            all_features.append(feats)

        if (i + 1) % 25 == 0 or (i + 1) == len(image_paths):
            print(f"Processadas {i + 1}/{len(image_paths)} imagens...")

    df = pd.DataFrame(all_features)
    output_csv = 'egg_features.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nExtração concluída com sucesso! {len(all_features)} ovos processados.")
    print(f"Arquivo gerado: '{output_csv}'.")

if __name__ == '__main__':
    main()
