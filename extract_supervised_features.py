import os
import glob
import cv2
import numpy as np
import pandas as pd

def load_image_safe(image_path, target_size=500):
    with open(image_path, "rb") as stream:
        bytes_array = bytearray(stream.read())
    np_array = np.asarray(bytes_array, dtype=np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = float(target_size) / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized

def extract_candling_features(image_path, label, class_name):
    """
    Extrai características ópticas de Transiluminação/Ovoscopia (Original Images(Eggs)).
    """
    img = load_image_safe(image_path, target_size=400)
    if img is None:
        return None

    h, w = img.shape[:2]
    total_pixels = float(h * w)
    b, g, r = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Na ovoscopia o fundo é preto; isolamos a região iluminada (canal vermelho > 25)
    _, mask = cv2.threshold(r, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    lit_pixels = float(np.sum(mask > 0))
    lit_area_ratio = lit_pixels / total_pixels

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and lit_pixels > 100:
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        perimeter = float(cv2.arcLength(c, True))
        hull_area = float(cv2.contourArea(cv2.convexHull(c)))
        solidity = area / hull_area if hull_area > 0 else 0.0
        circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        if len(c) >= 5:
            _, (ax1, ax2), _ = cv2.fitEllipse(c)
            major, minor = max(ax1, ax2), min(ax1, ax2)
            aspect_ratio = float(major / minor) if minor > 0 else 1.0
        else:
            aspect_ratio = 1.0
    else:
        area, perimeter, solidity, circularity, aspect_ratio = 0.0, 0.0, 0.0, 0.0, 1.0

    if lit_pixels > 0:
        mean_bgr, std_bgr = cv2.meanStdDev(img, mask=mask)
        mean_hsv, std_hsv = cv2.meanStdDev(hsv, mask=mask)
        eroded = cv2.erode(mask, kernel)
        eroded_area = float(np.sum(eroded > 0))
        if eroded_area > 0:
            edges = cv2.Canny(gray, 30, 90)
            internal_edge_density = float(np.sum((edges > 0) & (eroded > 0)) / eroded_area)
            bg_smooth = cv2.medianBlur(gray, 21)
            diff_spots = cv2.subtract(bg_smooth, gray)
            spot_density = float(np.sum((diff_spots > 15) & (eroded > 0)) / eroded_area)
        else:
            internal_edge_density, spot_density = 0.0, 0.0
    else:
        mean_bgr = np.zeros((3, 1))
        std_bgr = np.zeros((3, 1))
        mean_hsv = np.zeros((3, 1))
        std_hsv = np.zeros((3, 1))
        internal_edge_density, spot_density = 0.0, 0.0

    return {
        'filename': os.path.basename(image_path),
        'class_name': class_name,
        'label': label,
        'lit_area_ratio': lit_area_ratio,
        'area': area,
        'perimeter': perimeter,
        'solidity': solidity,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'mean_red': float(mean_bgr[2][0]),
        'mean_green': float(mean_bgr[1][0]),
        'mean_blue': float(mean_bgr[0][0]),
        'std_red': float(std_bgr[2][0]),
        'std_green': float(std_bgr[1][0]),
        'mean_hue': float(mean_hsv[0][0]),
        'mean_sat': float(mean_hsv[1][0]),
        'mean_val': float(mean_hsv[2][0]),
        'std_val': float(std_hsv[2][0]),
        'internal_edge_density': internal_edge_density,
        'spot_density': spot_density
    }

def extract_kaggle_features(image_path, label, class_name):
    """
    Extrai características de ovos brancos em cenários abertos (Kaggle Dataset).
    Foca no isolamento da casca clara central, fraturas (bordas internas e buracos escuros) e geometria.
    """
    img = load_image_safe(image_path, target_size=500)
    if img is None:
        return None

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ovos brancos têm alta luminância (V alto) e baixa saturação (S baixo)
    # Criamos um score de "brancura": V - S
    v_chan = hsv[:, :, 2].astype(np.int16)
    s_chan = hsv[:, :, 1].astype(np.int16)
    whiteness = np.clip(v_chan - s_chan, 0, 255).astype(np.uint8)

    blur = cv2.GaussianBlur(whiteness, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Seleciona contorno relevante mais próximo do centro
    cy_img, cx_img = h / 2.0, w / 2.0
    best_c = max(contours, key=lambda c: cv2.contourArea(c) / (1.0 + 0.05 * np.hypot(
        (cv2.moments(c)['m10']/cv2.moments(c)['m00'] if cv2.moments(c)['m00'] > 0 else cx_img) - cx_img,
        (cv2.moments(c)['m01']/cv2.moments(c)['m00'] if cv2.moments(c)['m00'] > 0 else cy_img) - cy_img
    )))

    area = float(cv2.contourArea(best_c))
    perimeter = float(cv2.arcLength(best_c, True))
    hull = cv2.convexHull(best_c)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0
    circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    if len(best_c) >= 5:
        _, (ax1, ax2), _ = cv2.fitEllipse(best_c)
        major, minor = max(ax1, ax2), min(ax1, ax2)
        aspect_ratio = float(major / minor) if minor > 0 else 1.0
    else:
        aspect_ratio = 1.0

    egg_mask = np.zeros_like(gray)
    cv2.drawContours(egg_mask, [best_c], -1, 255, -1)
    hull_mask = np.zeros_like(gray)
    cv2.drawContours(hull_mask, [hull], -1, 255, -1)

    mean_bgr, std_bgr = cv2.meanStdDev(img, mask=egg_mask)
    mean_hsv, std_hsv = cv2.meanStdDev(hsv, mask=egg_mask)

    # Dentro do Convex Hull do ovo, ovos quebrados têm cavidades escuras e muitas bordas de fratura
    hull_pixels = float(np.sum(hull_mask > 0))
    if hull_pixels > 0:
        edges = cv2.Canny(gray, 50, 150)
        fracture_edge_density = float(np.sum((edges > 0) & (hull_mask > 0)) / hull_pixels)
        dark_cavity_ratio = float(np.sum((gray < 100) & (hull_mask > 0)) / hull_pixels)
    else:
        fracture_edge_density, dark_cavity_ratio = 0.0, 0.0

    return {
        'filename': os.path.basename(image_path),
        'class_name': class_name,
        'label': label,
        'area': area,
        'perimeter': perimeter,
        'solidity': solidity,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'mean_red': float(mean_bgr[2][0]),
        'mean_green': float(mean_bgr[1][0]),
        'mean_blue': float(mean_bgr[0][0]),
        'std_red': float(std_bgr[2][0]),
        'mean_hue': float(mean_hsv[0][0]),
        'mean_sat': float(mean_hsv[1][0]),
        'mean_val': float(mean_hsv[2][0]),
        'std_val': float(std_hsv[2][0]),
        'fracture_edge_density': fracture_edge_density,
        'dark_cavity_ratio': dark_cavity_ratio
    }

def main():
    # 1. Extrair de Original Images(Eggs) - Ovoscopia
    candling_rows = []
    candling_classes = [
        (0, 'Good_Egg', os.path.join('Original Images(Eggs)', 'Good Eggs')),
        (1, 'Bad_Egg', os.path.join('Original Images(Eggs)', 'Bad Eggs'))
    ]
    print("=== Extraindo features de Original Images(Eggs) (Ovoscopia) ===")
    for label, cname, folder in candling_classes:
        paths = sorted(glob.glob(os.path.join(folder, '*.jpg')))
        for p in paths:
            feats = extract_candling_features(p, label, cname)
            if feats:
                candling_rows.append(feats)
        print(f"  Concluído {cname}: {len(paths)} imagens.")

    df_candling = pd.DataFrame(candling_rows)
    df_candling.to_csv('candling_features.csv', index=False)
    print(f"Salvo 'candling_features.csv' com {len(df_candling)} registros.\n")

    # 2. Extrair de Kaggle Dataset - Ovos Brancos Damaged vs Not Damaged
    kaggle_rows = []
    kaggle_classes = [
        (0, 'Not_Damaged', os.path.join('Kaggle Dataset', 'Eggs Classification', 'Not Damaged')),
        (1, 'Damaged', os.path.join('Kaggle Dataset', 'Eggs Classification', 'Damaged'))
    ]
    print("=== Extraindo features de Kaggle Dataset (Damaged vs Not Damaged) ===")
    for label, cname, folder in kaggle_classes:
        paths = sorted(glob.glob(os.path.join(folder, '*.jpg')))
        for p in paths:
            feats = extract_kaggle_features(p, label, cname)
            if feats:
                kaggle_rows.append(feats)
        print(f"  Concluído {cname}: {len(paths)} imagens.")

    df_kaggle = pd.DataFrame(kaggle_rows)
    df_kaggle.to_csv('kaggle_features.csv', index=False)
    print(f"Salvo 'kaggle_features.csv' com {len(df_kaggle)} registros.")

if __name__ == '__main__':
    main()
