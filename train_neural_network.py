import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def main():
    features_csv = 'egg_features.csv'
    outliers_csv = 'outliers_summary.csv'

    if not os.path.exists(features_csv):
        print(f"Erro: '{features_csv}' não encontrado. Execute 'extract_features.py' primeiro.")
        return

    df = pd.read_csv(features_csv)
    # Filtra apenas imagens que existem atualmente na pasta Dataset/
    existing_files = set(os.listdir('Dataset')) if os.path.exists('Dataset') else set(df['filename'])
    df = df[df['filename'].isin(existing_files)].reset_index(drop=True)

    # Carrega os rótulos de anomalias gerados para o Dataset autoral
    if os.path.exists(outliers_csv):
        out_df = pd.read_csv(outliers_csv)
        outlier_set = set(out_df['filename'])
    else:
        outlier_set = set()

    df['label'] = df['filename'].apply(lambda fn: 1 if fn in outlier_set else 0)
    df['class_name'] = df['label'].apply(lambda l: 'Anomalo' if l == 1 else 'Normal')

    feature_cols = [
        'area', 'perimeter', 'solidity', 'circularity', 'aspect_ratio', 'eccentricity',
        'mean_blue', 'mean_green', 'mean_red', 'std_red',
        'mean_hue', 'mean_sat', 'mean_val', 'std_val',
        'internal_edge_density', 'spot_density', 'texture_roughness'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X_raw = df[feature_cols].values
    y = df['label'].values

    scaler = RobustScaler()
    X = scaler.fit_transform(X_raw)

    print("=================================================================")
    print("  TREINAMENTO DE REDE NEURAL NO SEU DATASET AUTORAL (Dataset/)")
    print("=================================================================")
    print(f"Total de ovos no Dataset atual : {len(df)}")
    print(f"  • Ovos Normais  (Classe 0)   : {np.sum(y == 0)}")
    print(f"  • Ovos Anômalos (Classe 1)   : {np.sum(y == 1)}")
    print(f"Features de entrada na Rede    : {len(feature_cols)} neurônios\n")

    # =========================================================================
    # 1. REDE NEURAL AUTOENCODER (Não-Supervisionada - Reconstrução de Padrão)
    # =========================================================================
    print("--- [1/2] Treinando Rede Neural Autoencoder (Arquitetura 17 -> 10 -> 4 -> 10 -> 17) ---")
    X_normal = X[y == 0]
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(10, 4, 10),
        activation='relu',
        solver='adam',
        alpha=0.01,
        max_iter=1500,
        random_state=42
    )
    # Treina o Autoencoder para reconstruir os ovos normais
    autoencoder.fit(X_normal, X_normal)

    # Calcula o Erro de Reconstrução (MSE por ovo) em todo o dataset
    X_reconstructed = autoencoder.predict(X)
    reconstruction_errors = np.mean((X - X_reconstructed) ** 2, axis=1)
    df['autoencoder_mse'] = reconstruction_errors

    ae_auc = roc_auc_score(y, reconstruction_errors)
    # Limiar no percentil 85 dos erros de reconstrução
    ae_thresh = np.percentile(reconstruction_errors[y == 0], 90)
    ae_preds = (reconstruction_errors > ae_thresh).astype(int)
    print(f"  • Erro médio de reconstrução (Ovos Normais)  : {reconstruction_errors[y == 0].mean():.4f}")
    print(f"  • Erro médio de reconstrução (Ovos Anômalos) : {reconstruction_errors[y == 1].mean():.4f}")
    print(f"  • ROC-AUC do Autoencoder Não-Supervisionado  : {ae_auc * 100:.2f}%\n")

    # =========================================================================
    # 2. REDE NEURAL MLP CLASSIFICADORA (Supervisionada - 5-Fold CV)
    # =========================================================================
    print("--- [2/2] Treinando Rede Neural MLP Classificadora (Arquitetura 17 -> 64 -> 32 -> 1) ---")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.05,
        learning_rate_init=0.005,
        max_iter=1000,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(mlp, X, y, cv=cv, method='predict')
    y_prob_cv = cross_val_predict(mlp, X, y, cv=cv, method='predict_proba')[:, 1]

    acc = accuracy_score(y, y_pred_cv)
    prec = precision_score(y, y_pred_cv, zero_division=0)
    rec = recall_score(y, y_pred_cv, zero_division=0)
    f1 = f1_score(y, y_pred_cv, zero_division=0)
    auc = roc_auc_score(y, y_prob_cv)
    cm = confusion_matrix(y, y_pred_cv)

    print(f"Resultados da Validação Cruzada (5-Fold Stratified CV):")
    print(f"  • Acurácia  : {acc * 100:.2f}%")
    print(f"  • Precisão  : {prec * 100:.2f}%")
    print(f"  • Recall    : {rec * 100:.2f}%")
    print(f"  • F1-Score  : {f1 * 100:.2f}%")
    print(f"  • ROC-AUC   : {auc * 100:.2f}%\n")

    print("Matriz de Confusão (5-Fold CV):")
    print(f"  [ Verdadeiro Normal : {cm[0,0]:3d} | Falso Positivo    : {cm[0,1]:3d} ]")
    print(f"  [ Falso Negativo    : {cm[1,0]:3d} | Verdadeiro Anômalo: {cm[1,1]:3d} ]\n")

    # Treina modelo final com todos os dados para salvar
    mlp.fit(X, y)
    df['nn_probability'] = mlp.predict_proba(X)[:, 1]
    df['nn_prediction'] = mlp.predict(X)

    # Salva os resultados da rede neural em CSV
    df[['filename', 'class_name', 'label', 'nn_prediction', 'nn_probability', 'autoencoder_mse']].to_csv(
        'nn_predictions.csv', index=False
    )

    # Salva os modelos treinados e o scaler
    model_bundle = {
        'mlp_classifier': mlp,
        'autoencoder': autoencoder,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
    with open('egg_neural_network.pkl', 'wb') as f:
        pickle.dump(model_bundle, f)

    print("Arquivos gerados com sucesso:")
    print("  1. 'egg_neural_network.pkl' (Pesos da Rede Neural MLP + Autoencoder + Scaler)")
    print("  2. 'nn_predictions.csv' (Probabilidades e Erros de Reconstrução de cada ovo)")

if __name__ == '__main__':
    main()
