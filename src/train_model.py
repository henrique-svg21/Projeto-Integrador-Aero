"""
Pipeline de Treinamento Preditivo Multiclasse, Detecção de Anomalias e MLOps.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

from preprocessing import engenharia_features_stream, saneamento_outliers_e_ausentes


def main():
    # 1. Carregamento dos dados
    data_path = os.path.join("data", "processed", "dados_sensores_tratados.csv")
    if not os.path.exists(data_path):
        raw_path = os.path.join("data", "raw", "dados_sensores_aeromotion.csv")
        if not os.path.exists(raw_path):
            raw_path = "dados_sensores_aeromotion.csv"
        df = pd.read_csv(raw_path)
        df = saneamento_outliers_e_ausentes(df)
        df = engenharia_features_stream(df)
    else:
        df = pd.read_csv(data_path)

    # 2. Divisão Temporal (70% Treino e 30% Teste)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    n_samples = len(df)
    train_size = int(n_samples * 0.7)
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()

    feature_cols = [
        'modo_operacao', 'horas_operacao_acumuladas', 'temperatura_mancal_c',
        'vibracao_rms_mms', 'pressao_oleo_bar', 'rotacao_rpm',
        'emissao_acustica_db', 'corrente_eletrica_amp', 'vibracao_mean_1h',
        'vibracao_std_1h', 'temp_mean_1h', 'temp_std_1h',
        'vibracao_mean_6h', 'vibracao_std_6h', 'temp_mean_6h', 'temp_std_6h',
        'severidade_dinamica'
    ]
    target_col = 'tipo_falha'

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]

    # 3. Pipeline de Transformação (RobustScaler e OneHotEncoder)
    cat_cols = ['modo_operacao']
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # 4. Modelagem Supervisionada Multiclasse
    # Modelo 1: Random Forest com pesos balanceados
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_trans, y_train)
    y_pred_rf = rf_model.predict(X_test_trans)

    # Modelo 2: XGBoost Classifier
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train_trans, y_train)
    y_pred_xgb = xgb_model.predict(X_test_trans)

    print("=======================================================")
    print("           DESEMPENHO: RANDOM FOREST                  ")
    print("=======================================================")
    print(classification_report(y_test, y_pred_rf, digits=4))
    print("Matriz de Confusão RF (4x4):")
    print(confusion_matrix(y_test, y_pred_rf))

    print("\n=======================================================")
    print("              DESEMPENHO: XGBOOST                     ")
    print("=======================================================")
    print(classification_report(y_test, y_pred_xgb, digits=4))
    print("Matriz de Confusão XGBoost (4x4):")
    print(confusion_matrix(y_test, y_pred_xgb))

    # 5. Modelo Não Supervisionado: Isolation Forest (treinado em dados 100% normais)
    X_train_normal = X_train_trans[y_train == 0]
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.03,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_normal)

    # 6. Exportação de Modelos e Artefatos de MLOps
    models_dir = os.path.join("models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    joblib.dump(rf_model, os.path.join(models_dir, "best_model.joblib"))
    joblib.dump(iso_forest, os.path.join(models_dir, "isolation_forest.joblib"))
    print("\nArtefatos de inferência salvos com sucesso na pasta `models/`.")


if __name__ == "__main__":
    main()