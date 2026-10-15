import os
import shutil
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

def detect_univariate_iqr(df, column, condition='high', k=1.5):
    """
    Detecta outliers usando a regra clássica de amplitude interquartil (Tukey's IQR).
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    if condition == 'high':
        threshold = q3 + (k * iqr)
        is_outlier = df[column] > threshold
    elif condition == 'low':
        threshold = q1 - (k * iqr)
        is_outlier = df[column] < threshold
    else:
        threshold_low = q1 - (k * iqr)
        threshold_high = q3 + (k * iqr)
        is_outlier = (df[column] < threshold_low) | (df[column] > threshold_high)
        threshold = (threshold_low, threshold_high)

    return is_outlier, threshold

def main():
    csv_file = 'egg_features.csv'
    dataset_dir = 'Dataset'
    outlier_dir = 'Outliers_Encontrados'

    if not os.path.exists(csv_file):
        print(f"Erro: Arquivo '{csv_file}' não encontrado. Execute 'extract_features.py' primeiro.")
        return

    df = pd.read_csv(csv_file)
    print(f"Carregadas características de {len(df)} ovos.")

    # Dicionário para armazenar os motivos de cada arquivo
    outlier_reasons = {fn: [] for fn in df['filename']}

    # 1. DETECÇÃO UNIVARIADA INTERPRETÁVEL (IQR)
    checks = [
        ('spot_density', 'high', 1.5, 'MANCHAS_OU_SARDAS'),
        ('internal_edge_density', 'high', 1.5, 'TRINCA_OU_TEXTURA_IRREGULAR'),
        ('aspect_ratio', 'high', 2.0, 'FORMATO_MUITO_COMPRIDO'),
        ('solidity', 'low', 2.0, 'DEFORMACAO_CONTORNO'),
        ('mean_sat', 'low', 2.0, 'PIGMENTACAO_PALIDA'),
        ('mean_sat', 'high', 2.0, 'HIPERPIGMENTADO'),
        ('area', 'low', 2.5, 'TAMANHO_MINIATURA')
    ]

    print("\n--- Analisando anomalias univariadas via IQR (Tukey) ---")
    for col, cond, factor, label in checks:
        if col not in df.columns:
            continue
        mask, thresh = detect_univariate_iqr(df, col, condition=cond, k=factor)
        outliers_col = df[mask]['filename'].tolist()
        for fn in outliers_col:
            val = df.loc[df['filename'] == fn, col].values[0]
            outlier_reasons[fn].append(f"{label} ({col}={val:.3f})")
        if outliers_col:
            print(f"  [{label}]: {len(outliers_col)} ovos encontrados.")

    # 2. DETECÇÃO MULTIVARIADA VIA ISOLATION FOREST
    feature_cols = [
        'area', 'aspect_ratio', 'solidity', 'circularity',
        'mean_hue', 'mean_sat', 'mean_val',
        'internal_edge_density', 'spot_density', 'texture_roughness'
    ]
    present_cols = [c for c in feature_cols if c in df.columns]

    scaler = RobustScaler()
    X = scaler.fit_transform(df[present_cols])

    # Isolation Forest com contaminação estimada em 5%
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    preds = iso_forest.fit_predict(X)
    scores = iso_forest.decision_function(X) # valores menores = mais anômalo

    df['anomaly_score'] = scores
    df['iso_forest_outlier'] = preds == -1

    iso_outliers = df[df['iso_forest_outlier']]['filename'].tolist()
    print(f"\n--- Analisando anomalias multivariadas via Isolation Forest ---")
    print(f"  Ovos atípicos identificados pelo modelo global: {len(iso_outliers)}")
    for fn in iso_outliers:
        outlier_reasons[fn].append("ANOMALIA_GLOBAL_ISOLATION_FOREST")

    # 3. PREPARAR PASTA DE DESTINO E RELATÓRIO
    if not os.path.exists(outlier_dir):
        os.makedirs(outlier_dir)

    # Limpar pasta anterior
    for f in os.listdir(outlier_dir):
        try:
            os.remove(os.path.join(outlier_dir, f))
        except (PermissionError, OSError):
            pass

    flagged_eggs = [fn for fn, reasons in outlier_reasons.items() if len(reasons) > 0]
    print(f"\n==========================================")
    print(f"TOTAL DE OVOS ANÔMALOS ENCONTRADOS: {len(flagged_eggs)} de {len(df)}")
    print(f"==========================================")

    records = []
    for fn in flagged_eggs:
        reasons = outlier_reasons[fn]
        reasons_str = "; ".join(reasons)
        score = df.loc[df['filename'] == fn, 'anomaly_score'].values[0]
        records.append({
            'filename': fn,
            'reasons': reasons_str,
            'num_reasons': len(reasons),
            'anomaly_score': score
        })

        # Copiar para a pasta de outliers apenas se o arquivo existir no Dataset
        src = os.path.join(dataset_dir, fn)
        primary_reason = reasons[0].split()[0]
        dst = os.path.join(outlier_dir, f"{primary_reason}___{fn}")
        if os.path.exists(src):
            try:
                shutil.copy2(src, dst)
            except (PermissionError, OSError):
                pass

        print(f"  • {fn}: {reasons_str}")

    # Salvar relatório consolidado em CSV
    report_df = pd.DataFrame(records)
    if not report_df.empty:
        report_df.sort_values(by=['num_reasons', 'anomaly_score'], ascending=[False, True], inplace=True)
        report_df.to_csv('outliers_summary.csv', index=False)
        print(f"\nRelatório detalhado salvo em 'outliers_summary.csv'.")
    print(f"Imagens de anomalias copiadas para '{outlier_dir}'.")

if __name__ == '__main__':
    main()
