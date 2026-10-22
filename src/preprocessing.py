"""
Módulo de pré-processamento de séries temporais para AeroMotion Dynamics S.A.
"""
from typing import List
import numpy as np
import pandas as pd


def saneamento_outliers_e_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """Substitui ruídos físicos causados por EMI e interpola valores ausentes."""
    df_clean = df.copy()
    if 'timestamp' in df_clean.columns:
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        df_clean = df_clean.sort_values(by=['id_equipamento', 'timestamp']).reset_index(drop=True)

    # Limites operacionais físicos
    df_clean.loc[df_clean['temperatura_mancal_c'] > 150.0, 'temperatura_mancal_c'] = np.nan
    df_clean.loc[df_clean['vibracao_rms_mms'] < 0.0, 'vibracao_rms_mms'] = np.nan

    num_cols = [
        'temperatura_mancal_c', 'vibracao_rms_mms', 'pressao_oleo_bar',
        'rotacao_rpm', 'emissao_acustica_db', 'corrente_eletrica_amp'
    ]

    df_interpolado: List[pd.DataFrame] = []
    for _, sub in df_clean.groupby('id_equipamento', as_index=False):
        sub = sub.copy()
        if 'timestamp' in sub.columns:
            sub = sub.sort_values('timestamp')
        sub[num_cols] = sub[num_cols].interpolate(method='linear').bfill().ffill()
        df_interpolado.append(sub)

    return pd.concat(df_interpolado, ignore_index=True)


def engenharia_features_stream(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera features derivadas para dados em lote ou telemetria contínua.
    """
    df_out = df.copy()
    if 'timestamp' in df_out.columns:
        df_out['timestamp'] = pd.to_datetime(df_out['timestamp'])
        df_out = df_out.sort_values(['id_equipamento', 'timestamp'])

    frames: List[pd.DataFrame] = []
    for _, sub in df_out.groupby('id_equipamento', as_index=False):
        sub = sub.copy()
        if 'timestamp' in sub.columns:
            sub = sub.set_index('timestamp')
            sub['vibracao_mean_1h'] = sub['vibracao_rms_mms'].rolling('1h', min_periods=1).mean()
            sub['vibracao_std_1h'] = sub['vibracao_rms_mms'].rolling('1h', min_periods=1).std().fillna(0)
            sub['temp_mean_1h'] = sub['temperatura_mancal_c'].rolling('1h', min_periods=1).mean()
            sub['temp_std_1h'] = sub['temperatura_mancal_c'].rolling('1h', min_periods=1).std().fillna(0)
            sub['vibracao_mean_6h'] = sub['vibracao_rms_mms'].rolling('6h', min_periods=1).mean()
            sub['vibracao_std_6h'] = sub['vibracao_rms_mms'].rolling('6h', min_periods=1).std().fillna(0)
            sub['temp_mean_6h'] = sub['temperatura_mancal_c'].rolling('6h', min_periods=1).mean()
            sub['temp_std_6h'] = sub['temperatura_mancal_c'].rolling('6h', min_periods=1).std().fillna(0)
            sub = sub.reset_index()
        else:
            # Fallback para amostras estáticas
            sub['vibracao_mean_1h'] = sub['vibracao_rms_mms']
            sub['vibracao_std_1h'] = 0.0
            sub['temp_mean_1h'] = sub['temperatura_mancal_c']
            sub['temp_std_1h'] = 0.0
            sub['vibracao_mean_6h'] = sub['vibracao_rms_mms']
            sub['vibracao_std_6h'] = 0.0
            sub['temp_mean_6h'] = sub['temperatura_mancal_c']
            sub['temp_std_6h'] = 0.0

        sub['severidade_dinamica'] = sub['vibracao_rms_mms'] / sub['rotacao_rpm']
        frames.append(sub)

    return pd.concat(frames, ignore_index=True)