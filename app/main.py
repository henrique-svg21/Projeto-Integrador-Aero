"""
Microsserviço RESTful e Portal Analítico Industrial (Design System WEGxpert / WEG).
AeroMotion Dynamics S.A. — Plataforma Integrada de IA, Telemetria e Visão Computacional.
"""
from contextlib import asynccontextmanager
from enum import Enum
import os
from pathlib import Path
import sys
from typing import Any, Dict, List
import cv2
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.vision_inspection import IndustrialVisionInspector

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
ISO_PATH = MODEL_DIR / "isolation_forest.joblib"

FALLA_MAP = {
    0: "Normal",
    1: "Desgaste_Rolamento",
    2: "Desbalanceamento_Rotor",
    3: "Cavitacao_Superaquecimento"
}

artifacts: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
        artifacts["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
        artifacts["classifier"] = joblib.load(MODEL_PATH)
        if ISO_PATH.exists():
            artifacts["anomaly_detector"] = joblib.load(ISO_PATH)
        print("-> Modelos de IA carregados com sucesso em memória.")
    else:
        print("-> AVISO: Modelos nao encontrados em models/. Execute src/train_model.py.")
    yield
    artifacts.clear()


app = FastAPI(
    title="AeroMotion Dynamics - WEGxpert Industrial AI Portal",
    version="2.0.0",
    lifespan=lifespan
)


class ModoOperacaoEnum(str, Enum):
    cruzeiro = "Cruzeiro"
    carga_media = "Carga_Media"
    carga_alta = "Carga_Alta"


class TelemetriaLeitura(BaseModel):
    id_equipamento: str = Field(..., examples=["TURB-101"])
    modo_operacao: ModoOperacaoEnum = Field(..., examples=["Carga_Alta"])
    horas_operacao_acumuladas: float = Field(..., ge=0.0, examples=[1250.5])
    temperatura_mancal_c: float = Field(..., ge=-20.0, le=250.0, examples=[78.4])
    vibracao_rms_mms: float = Field(..., ge=0.0, le=50.0, examples=[3.2])
    pressao_oleo_bar: float = Field(..., ge=0.0, le=15.0, examples=[4.3])
    rotacao_rpm: float = Field(..., ge=500.0, le=6000.0, examples=[3450.0])
    emissao_acustica_db: float = Field(..., ge=10.0, le=150.0, examples=[58.2])
    corrente_eletrica_amp: float = Field(..., ge=0.0, le=100.0, examples=[38.1])


class PredicaoResposta(BaseModel):
    id_equipamento: str
    diagnostico_codigo: int
    diagnostico_descricao: str
    probabilidades: Dict[str, float]
    alerta_anomalia_desconhecida: bool


def preparar_features(leituras: List[TelemetriaLeitura]) -> pd.DataFrame:
    registros = []
    for l in leituras:
        reg = l.model_dump()
        reg['modo_operacao'] = l.modo_operacao.value
        reg['vibracao_mean_1h'] = reg['vibracao_rms_mms']
        reg['vibracao_std_1h'] = 0.0
        reg['temp_mean_1h'] = reg['temperatura_mancal_c']
        reg['temp_std_1h'] = 0.0
        reg['vibracao_mean_6h'] = reg['vibracao_rms_mms']
        reg['vibracao_std_6h'] = 0.0
        reg['temp_mean_6h'] = reg['temperatura_mancal_c']
        reg['temp_std_6h'] = 0.0
        reg['severidade_dinamica'] = reg['vibracao_rms_mms'] / reg['rotacao_rpm']
        registros.append(reg)
    return pd.DataFrame(registros)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    model_loaded = "classifier" in artifacts
    return {
        "status": "online" if model_loaded else "degraded",
        "service": "AeroMotion Predictive Inference API",
        "model_version": "2.0.0-wegxpert-ui",
        "model_loaded": model_loaded
    }


@app.get("/analytics/summary")
def get_analytics_summary():
    """Retorna as métricas reais consolidadas do dataset tratado que alimentam a IA."""
    return {
        "total_amostras": 12000,
        "outliers_emi_corrigidos": 103,
        "nans_interpolados": 1573,
        "acuracia_xgboost": 99.67,
        "recall_cavitacao": 100.0,
        "distribuicao_classes": {
            "labels": ["0: Normal (88%)", "1: Desgaste Rolamento (5%)", "2: Desbalanceamento (4%)", "3: Cavitação/Superaq. (3%)"],
            "counts": [10560, 600, 480, 360]
        },
        "assinatura_fisica_falhas": {
            "classes": ["Normal", "Desgaste Rolamento", "Desbalanceamento", "Cavitação/Superaq."],
            "temperatura_c": [62.98, 62.82, 63.44, 97.23],
            "vibracao_mms": [2.77, 9.25, 10.48, 2.83],
            "emissao_acustica_db": [53.47, 80.01, 53.87, 53.54],
            "pressao_oleo_bar": [4.30, 4.29, 4.31, 2.07]
        },
        "regimes_operacao": {
            "regimes": ["Cruzeiro (2.400 RPM)", "Carga Média (3.000 RPM)", "Carga Alta (3.500 RPM)"],
            "counts": [2381, 6024, 3595],
            "temp_media": [53.40, 62.98, 72.78],
            "vib_media": [2.44, 3.18, 4.43]
        },
        "importancia_features": {
            "features": [
                "Severidade Dinâmica (Vib/RPM)",
                "Vibração RMS (mm/s)",
                "Emissão Acústica (dB)",
                "Temperatura Mancal (°C)",
                "Pressão do Óleo (Bar)",
                "Vibração Média Móvel (1h)",
                "Temp. Média Móvel (6h)"
            ],
            "pesos": [28.4, 22.1, 18.5, 13.8, 8.9, 4.8, 3.5]
        }
    }


@app.post("/predict", response_model=PredicaoResposta)
def predict_single(telemetria: TelemetriaLeitura):
    if "classifier" not in artifacts or "preprocessor" not in artifacts:
        raise HTTPException(status_code=503, detail="Modelos nao carregados.")

    df_in = preparar_features([telemetria])
    feature_cols = [
        'modo_operacao', 'horas_operacao_acumuladas', 'temperatura_mancal_c',
        'vibracao_rms_mms', 'pressao_oleo_bar', 'rotacao_rpm',
        'emissao_acustica_db', 'corrente_eletrica_amp', 'vibracao_mean_1h',
        'vibracao_std_1h', 'temp_mean_1h', 'temp_std_1h',
        'vibracao_mean_6h', 'vibracao_std_6h', 'temp_mean_6h', 'temp_std_6h',
        'severidade_dinamica'
    ]
    X_trans = artifacts["preprocessor"].transform(df_in[feature_cols])
    pred_class = int(artifacts["classifier"].predict(X_trans)[0])
    probs = artifacts["classifier"].predict_proba(X_trans)[0]
    prob_dict = {FALLA_MAP[i]: float(round(p, 4)) for i, p in enumerate(probs)}

    is_anomaly = False
    if "anomaly_detector" in artifacts:
        iso_pred = artifacts["anomaly_detector"].predict(X_trans)[0]
        is_anomaly = bool(iso_pred == -1)

    return PredicaoResposta(
        id_equipamento=telemetria.id_equipamento,
        diagnostico_codigo=pred_class,
        diagnostico_descricao=FALLA_MAP[pred_class],
        probabilidades=prob_dict,
        alerta_anomalia_desconhecida=is_anomaly
    )


@app.post("/predict-batch", response_model=List[PredicaoResposta])
def predict_batch(telemetrias: List[TelemetriaLeitura]):
    if "classifier" not in artifacts or "preprocessor" not in artifacts:
        raise HTTPException(status_code=503, detail="Modelos nao carregados.")

    df_in = preparar_features(telemetrias)
    feature_cols = [
        'modo_operacao', 'horas_operacao_acumuladas', 'temperatura_mancal_c',
        'vibracao_rms_mms', 'pressao_oleo_bar', 'rotacao_rpm',
        'emissao_acustica_db', 'corrente_eletrica_amp', 'vibracao_mean_1h',
        'vibracao_std_1h', 'temp_mean_1h', 'temp_std_1h',
        'vibracao_mean_6h', 'vibracao_std_6h', 'temp_mean_6h', 'temp_std_6h',
        'severidade_dinamica'
    ]
    X_trans = artifacts["preprocessor"].transform(df_in[feature_cols])
    preds = artifacts["classifier"].predict(X_trans)
    probs = artifacts["classifier"].predict_proba(X_trans)

    iso_preds = [1] * len(preds)
    if "anomaly_detector" in artifacts:
        iso_preds = artifacts["anomaly_detector"].predict(X_trans)

    respostas = []
    for idx, row in enumerate(telemetrias):
        c = int(preds[idx])
        p_dict = {FALLA_MAP[i]: float(round(probs[idx][i], 4)) for i in range(len(FALLA_MAP))}
        respostas.append(
            PredicaoResposta(
                id_equipamento=row.id_equipamento,
                diagnostico_codigo=c,
                diagnostico_descricao=FALLA_MAP[c],
                probabilidades=p_dict,
                alerta_anomalia_desconhecida=bool(iso_preds[idx] == -1)
            )
        )
    return respostas


@app.post("/inspect/compare")
async def inspect_compare_endpoint(
    file_ref: UploadFile = File(...),
    file_test: UploadFile = File(...),
    min_circularidade: float = Form(0.70),
    max_desgaste_pct: float = Form(10.0),
    diff_threshold: int = Form(35),
    canny_high: int = Form(150)
):
    b_ref = await file_ref.read()
    b_test = await file_test.read()

    img_ref = cv2.imdecode(np.frombuffer(b_ref, np.uint8), cv2.IMREAD_COLOR)
    img_test = cv2.imdecode(np.frombuffer(b_test, np.uint8), cv2.IMREAD_COLOR)

    if img_ref is None or img_test is None:
        raise HTTPException(status_code=400, detail="Imagem enviada invalida.")

    aprovado, relatorio = IndustrialVisionInspector.comparar_componentes(
        img_ref=img_ref,
        img_test=img_test,
        min_circularidade=min_circularidade,
        max_desgaste_pct=max_desgaste_pct,
        diff_threshold=diff_threshold,
        canny_low=int(canny_high / 3),
        canny_high=canny_high
    )
    return {
        "status_geral": "APROVADO" if aprovado else "REJEITADO",
        "nome_ref": file_ref.filename,
        "nome_test": file_test.filename,
        "relatorio": relatorio
    }


@app.post("/inspect/thermal")
async def inspect_thermal_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Imagem invalida.")
    critico, metricas = IndustrialVisionInspector.inspecao_termica(img)
    return {
        "arquivo": file.filename,
        "alarme_superaquecimento": critico,
        "metricas": metricas
    }


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AeroMotion | Expert Industrial AI & Vision Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --weg-navy-deep: #001633;
            --weg-navy: #00234b;
            --weg-blue: #00579d;
            --weg-cyan: #5bc2e7;
            --weg-light: #e6f2fa;
            --weg-card-bg: #072244;
            --weg-card-border: #133e70;
            --text-main: #f8fafc;
            --text-muted: #94b2d1;
        }

        html { scroll-behavior: smooth; }

        body {
            background-color: var(--weg-navy-deep);
            color: var(--text-main);
            font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            overflow-x: hidden;
        }

        /* Top Navbar Estilo Institucional WEG */
        .weg-navbar {
            background: linear-gradient(90deg, #001a38 0%, #002b5c 50%, #001a38 100%);
            border-bottom: 2px solid var(--weg-blue);
            padding: 12px 28px;
            position: sticky;
            top: 0;
            z-index: 1050;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.5);
        }

        .weg-logo-box {
            border: 2px solid #ffffff;
            padding: 3px 10px;
            font-weight: 900;
            letter-spacing: 2px;
            font-size: 1.25rem;
            color: #ffffff;
            display: inline-block;
            line-height: 1.1;
            margin-right: 14px;
        }

        .nav-anchor {
            color: #d0e4f7;
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            padding: 8px 14px;
            border-radius: 4px;
            transition: all 0.2s;
        }
        .nav-anchor:hover {
            color: var(--weg-cyan);
            background: rgba(91, 194, 231, 0.1);
        }

        /* Hero Banner Estilo WEGxpert */
        .weg-hero {
            position: relative;
            background: radial-gradient(circle at 80% 30%, rgba(0, 87, 157, 0.45) 0%, rgba(0, 22, 51, 0.96) 70%),
                        linear-gradient(135deg, #001c3d 0%, #003066 100%);
            border-bottom: 1px solid var(--weg-card-border);
            padding: 48px 32px;
            overflow: hidden;
        }
        .weg-hero::before {
            content: "";
            position: absolute;
            top: 0; right: 0; bottom: 0; left: 0;
            background-image: radial-gradient(rgba(91, 194, 231, 0.22) 1.5px, transparent 1.5px);
            background-size: 22px 22px;
            pointer-events: none;
        }

        .wegxpert-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(91, 194, 231, 0.14);
            border: 1px solid var(--weg-cyan);
            color: var(--weg-cyan);
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 14px;
        }

        .btn-weg-pill {
            border: 1.5px solid #ffffff;
            color: #ffffff;
            background: transparent;
            border-radius: 50px;
            padding: 9px 26px;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.25s;
            text-decoration: none;
            display: inline-block;
        }
        .btn-weg-pill:hover {
            background: var(--weg-cyan);
            border-color: var(--weg-cyan);
            color: #001633;
        }

        .btn-weg-action {
            background: linear-gradient(90deg, var(--weg-blue) 0%, #0077d6 100%);
            border: 1px solid var(--weg-cyan);
            color: #ffffff;
            font-weight: 700;
            border-radius: 6px;
            padding: 11px 20px;
            transition: all 0.25s;
        }
        .btn-weg-action:hover {
            background: var(--weg-cyan);
            color: #001633;
            box-shadow: 0 0 16px rgba(91, 194, 231, 0.5);
        }

        .section-block {
            padding: 44px 18px;
            border-bottom: 1px solid rgba(19, 62, 112, 0.6);
        }

        .section-header-tag {
            color: var(--weg-cyan);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            display: block;
            margin-bottom: 4px;
        }

        .section-title {
            font-weight: 800;
            font-size: 1.65rem;
            color: #ffffff;
            border-left: 5px solid var(--weg-cyan);
            padding-left: 14px;
            margin-bottom: 24px;
        }

        .weg-card {
            background-color: var(--weg-card-bg);
            border: 1px solid var(--weg-card-border);
            border-radius: 10px;
            padding: 22px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
            height: 100%;
        }

        .kpi-box {
            background: linear-gradient(145deg, #051b38 0%, #0a2952 100%);
            border: 1px solid #194b85;
            border-top: 3px solid var(--weg-cyan);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        .kpi-value {
            font-size: 1.7rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 2px;
        }
        .kpi-label {
            font-size: 0.78rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }

        .preview-box {
            height: 215px;
            border: 2px dashed #22548f;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            background: #031124;
        }
        .preview-box img { max-height: 100%; max-width: 100%; object-fit: contain; }

        .slider-label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #d0e4f7;
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }
        .form-range::-webkit-slider-thumb { background: var(--weg-cyan); }

        .form-control, .form-select {
            background-color: #031329 !important;
            border: 1px solid #1d4d84 !important;
            color: #ffffff !important;
        }
        .form-control:focus, .form-select:focus {
            border-color: var(--weg-cyan) !important;
            box-shadow: 0 0 0 0.2rem rgba(91, 194, 231, 0.25) !important;
        }

        .chart-container {
            position: relative;
            height: 265px;
            width: 100%;
        }

        .badge-heartbeat {
            font-size: 0.88rem;
            padding: 7px 16px;
            border-radius: 30px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
        }
        /* Pulso verde quando ativo */
        .badge-heartbeat.bg-success {
            animation: pulse-green 2s infinite ease-in-out;
        }
        /* Pulso vermelho rápido de alerta quando desconectado */
        .badge-heartbeat.bg-danger {
            animation: pulse-red 1s infinite ease-in-out;
        }

        @keyframes pulse-green {
            0%, 100% { box-shadow: 0 0 6px rgba(34, 197, 94, 0.4); }
            50% { box-shadow: 0 0 18px rgba(34, 197, 94, 0.9), 0 0 30px rgba(34, 197, 94, 0.3); }
        }
        @keyframes pulse-red {
            0%, 100% { box-shadow: 0 0 6px rgba(239, 68, 68, 0.5); }
            50% { box-shadow: 0 0 20px rgba(239, 68, 68, 1), 0 0 35px rgba(239, 68, 68, 0.4); }
        }

        /* LED indicador piscante */
        .status-led {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #ffffff;
            display: inline-block;
            box-shadow: 0 0 8px #ffffff;
            animation: led-blink 1.4s infinite ease-in-out;
        }
        @keyframes led-blink {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.35; transform: scale(0.75); }
        }

        footer {
            background-color: #000e21;
            border-top: 2px solid var(--weg-blue);
            padding: 28px 20px;
            color: var(--text-muted);
            font-size: 0.85rem;
        }
    </style>
</head>
<body>

    <!-- NAVBAR ESTILO WEG -->
    <nav class="weg-navbar d-flex flex-wrap justify-content-between align-items-center">
        <div class="d-flex align-items-center">
            <div class="weg-logo-box">AERO<span style="color: var(--weg-cyan);">MOTION</span></div>
            <div class="d-none d-md-block">
                <span class="fw-bold text-white d-block" style="font-size: 0.95rem; line-height: 1.1;">AeroMotion Dynamics S.A.</span>
                <small style="color: var(--weg-cyan); font-size: 0.72rem; letter-spacing: 0.5px;">DIVISÃO DE TURBINAS & PROPULSÃO • EXPERT AI</small>
            </div>
        </div>

        <div class="d-none d-xl-flex gap-2">
            <a href="#sec-analytics" class="nav-anchor">01. Inteligência de Dados</a>
            <a href="#sec-vision" class="nav-anchor">02. Metrologia Óptica (Canny)</a>
            <a href="#sec-thermal" class="nav-anchor">03. Termografia HSV</a>
            <a href="#sec-telemetry" class="nav-anchor">04. Predição IoT</a>
        </div>

        <div class="d-flex align-items-center gap-2">
            <span id="badgeEdgeStatus" class="badge bg-secondary badge-heartbeat">⌛ Verificando nó...</span>
            <a href="/docs" target="_blank" class="btn-weg-pill py-1 px-3" style="font-size: 0.8rem;">Swagger OpenAPI</a>
        </div>
    </nav>

    <!-- HERO BANNER ESTILO WEGXPERT -->
    <header class="weg-hero">
        <div class="container-fluid max-w-7xl">
            <div class="row align-items-center">
                <div class="col-lg-8">
                    <div class="wegxpert-badge">
                        <span style="width:10px; height:10px; background:var(--weg-cyan); display:inline-block;"></span>
                        PLATAFORMA INDUSTRIAL UNIFICADA • EDGE COMPUTING
                    </div>
                    <h1 class="display-6 fw-bold text-white mb-2">
                        Sua planta industrial não para de evoluir.<br>
                        <span style="color: var(--weg-cyan);">Nossa Inteligência Preditiva também não.</span>
                    </h1>
                    <p class="text-light opacity-75 mb-4" style="max-width: 720px; font-size: 1.02rem;">
                        Monitoramento contínuo das bancadas <strong>TURB-101 a TURB-104</strong> integrando saneamento de séries temporais, 
                        modelagem multiclasse <strong>XGBoost</strong>, detecção de anomalias <strong>Isolation Forest</strong> e inspeção óptica <strong>OpenCV</strong> em página única.
                    </p>
                    <div class="d-flex flex-wrap gap-3">
                        <a href="#sec-analytics" class="btn-weg-action text-decoration-none">Explorar Métricas da IA</a>
                        <a href="#sec-vision" class="btn-weg-action text-decoration-none">Ir para Inspeção de Peças</a>
                    </div>
                </div>
                <div class="col-lg-4 d-none d-lg-block text-end">
                    <div class="p-3 rounded-3" style="background: rgba(0, 22, 51, 0.75); border: 1px solid var(--weg-cyan); display: inline-block; text-align: left;">
                        <div class="d-flex align-items-center gap-2 mb-2">
                            <div style="width: 18px; height: 18px; background: #ffffff; display: inline-block;"></div>
                            <div style="width: 18px; height: 18px; background: var(--weg-cyan); display: inline-block; margin-left: -8px; margin-top: -8px;"></div>
                            <span class="fw-bold text-white ms-1">Expert Analytics</span>
                        </div>
                        <small class="d-block text-secondary">Bancadas Monitoradas: <strong class="text-white">4 Turbinas (IoT 10 min)</strong></small>
                        <small class="d-block text-secondary">Modelo Produtivo: <strong style="color: var(--weg-cyan);">XGBoost + Isolation Forest</strong></small>
                        <small class="d-block text-secondary">Recall Cavitação Crítica: <strong class="text-success">100,00%</strong></small>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <main class="container-fluid px-4">

        <!-- SEÇÃO 1: INTELIGÊNCIA DE DADOS & MÉTRICAS DA IA -->
        <section id="sec-analytics" class="section-block">
            <span class="section-header-tag">ETAPAS 1 & 2 • DIAGNÓSTICO DE SÉRIES TEMPORAIS E MLOPS</span>
            <h2 class="section-title">1. Inteligência de Dados & Métricas que Alimentam a IA</h2>

            <div class="row g-3 mb-4">
                <div class="col-6 col-md-4 col-xl-2">
                    <div class="kpi-box">
                        <div class="kpi-value">12.000</div>
                        <div class="kpi-label">Leituras Telemetria IoT</div>
                    </div>
                </div>
                <div class="col-6 col-md-4 col-xl-2">
                    <div class="kpi-box">
                        <div class="kpi-value" style="color: var(--weg-cyan);">103</div>
                        <div class="kpi-label">Ruídos EMI Filtrados</div>
                    </div>
                </div>
                <div class="col-6 col-md-4 col-xl-2">
                    <div class="kpi-box">
                        <div class="kpi-value">1.573</div>
                        <div class="kpi-label">NaNs Interpolados</div>
                    </div>
                </div>
                <div class="col-6 col-md-4 col-xl-2">
                    <div class="kpi-box">
                        <div class="kpi-value text-success">99,67%</div>
                        <div class="kpi-label">Acurácia XGBoost</div>
                    </div>
                </div>
                <div class="col-6 col-md-4 col-xl-2">
                    <div class="kpi-box">
                        <div class="kpi-value text-success">100%</div>
                        <div class="kpi-label">Recall em Cavitação</div>
                    </div>
                </div>
                <div class="col-6 col-md-4 col-xl-2">
                    <div class="kpi-box">
                        <div class="kpi-value" style="color: #fbbf24;">17</div>
                        <div class="kpi-label">Features Preditivas</div>
                    </div>
                </div>
            </div>

            <div class="row g-4">
                <div class="col-lg-6">
                    <div class="weg-card">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h6 class="fw-bold text-white mb-0">Assinatura Física Média por Modo de Falha (O que a IA Aprendeu)</h6>
                            <span class="badge bg-dark border border-secondary text-info">Pós-Saneamento</span>
                        </div>
                        <p class="small mb-3" style="color: var(--text-muted);">
                            Demonstra como cada avaria altera fisicamente os sensores: Desgaste eleva a emissão acústica (80 dB), Desbalanceamento dispara a vibração (10.5 mm/s) e Cavitação eleva a temperatura (97 °C).
                        </p>
                        <div class="chart-container">
                            <canvas id="chartFaultSignatures"></canvas>
                        </div>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="weg-card">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h6 class="fw-bold text-white mb-0">Relevância das Variáveis na Decisão da IA (Feature Importance)</h6>
                            <span class="badge bg-dark border border-secondary text-info">XGBoost Gain</span>
                        </div>
                        <p class="small mb-3" style="color: var(--text-muted);">
                            Peso percentual de cada atributo engenheirado na separação das falhas. Destaque para o indicador físico derivado <strong>Severidade Dinâmica (Vibração / RPM)</strong>.
                        </p>
                        <div class="chart-container">
                            <canvas id="chartFeatureImportance"></canvas>
                        </div>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="weg-card">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h6 class="fw-bold text-white mb-0">Comportamento Termomecânico por Regime Operacional</h6>
                            <span class="badge bg-dark border border-secondary text-info">3 Regimes</span>
                        </div>
                        <p class="small mb-3" style="color: var(--text-muted);">
                            Relação entre Cruzeiro (2.400 RPM), Carga Média (3.000 RPM) e Carga Alta (3.500 RPM) na elevação natural de temperatura e vibração das bancadas.
                        </p>
                        <div class="chart-container">
                            <canvas id="chartOperatingRegimes"></canvas>
                        </div>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="weg-card">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h6 class="fw-bold text-white mb-0">Distribuição de Classes & Desbalanceamento Severo (12.000 amostras)</h6>
                            <span class="badge bg-dark border border-secondary text-warning">12% Falhas Reais</span>
                        </div>
                        <p class="small mb-3" style="color: var(--text-muted);">
                            Proporção real do dataset industrial que motivou a ponderação balanceada de classes e o treinamento do detector não supervisionado Isolation Forest.
                        </p>
                        <div class="chart-container">
                            <canvas id="chartClassDistribution"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- SEÇÃO 2: METROLOGIA ÓPTICA COMPARATIVA (CANNY & TOLERÂNCIAS GD&T) -->
        <section id="sec-vision" class="section-block">
            <span class="section-header-tag">ETAPA 3.1 • VISÃO COMPUTACIONAL GEOMÉTRICA (OPENCV)</span>
            <h2 class="section-title">2. Inspeção Visual Comparativa de Peças & Tolerâncias GD&T</h2>

            <div class="row g-4">
                <div class="col-lg-5">
                    <div class="weg-card">
                        <h5 class="fw-bold mb-3" style="color: var(--weg-cyan);">1. Padrão de Referência vs. Peça em Inspeção</h5>

                        <div class="row g-3 mb-3">
                            <div class="col-md-6">
                                <label class="form-label small fw-bold">Peça Nominal (Referência):</label>
                                <input type="file" id="fileRef" class="form-control form-control-sm mb-2" accept="image/*">
                                <div class="preview-box" id="boxRef"><span class="text-secondary small">Aguardando padrão...</span></div>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-bold">Peça em Teste (Desgaste):</label>
                                <input type="file" id="fileTest" class="form-control form-control-sm mb-2" accept="image/*">
                                <div class="preview-box" id="boxTest"><span class="text-secondary small">Aguardando teste...</span></div>
                            </div>
                        </div>

                        <hr class="border-secondary my-3">
                        <h6 class="fw-bold text-white mb-3">⚙️ Parametrização de Tolerâncias de Engenharia (GD&T)</h6>

                        <div class="mb-3">
                            <div class="slider-label">
                                <span>Tolerância Máx. Desgaste Superficial:</span>
                                <span id="lblWear" style="color: var(--weg-cyan);" class="fw-bold">10%</span>
                            </div>
                            <input type="range" class="form-range" id="rngWear" min="1" max="60" step="1" value="10">
                            <small class="text-secondary d-block" style="font-size: 0.74rem;">Pás de alta rotação: 2-5% | Carcaças externas: 15-40%</small>
                        </div>

                        <div class="mb-3">
                            <div class="slider-label">
                                <span>Circularidade Mínima Admissível:</span>
                                <span id="lblCirc" style="color: var(--weg-cyan);" class="fw-bold">0.70</span>
                            </div>
                            <input type="range" class="form-range" id="rngCirc" min="0.40" max="0.95" step="0.02" value="0.70">
                        </div>

                        <div class="row g-2 mb-3">
                            <div class="col-6">
                                <div class="slider-label">
                                    <span>Filtro de Luz/Ruído:</span>
                                    <span id="lblDiff" style="color: var(--weg-cyan);" class="fw-bold">35</span>
                                </div>
                                <input type="range" class="form-range" id="rngDiff" min="15" max="90" step="1" value="35">
                            </div>
                            <div class="col-6">
                                <div class="slider-label">
                                    <span>Limiar Canny High:</span>
                                    <span id="lblCanny" style="color: var(--weg-cyan);" class="fw-bold">150</span>
                                </div>
                                <input type="range" class="form-range" id="rngCanny" min="80" max="250" step="5" value="150">
                            </div>
                        </div>

                        <button id="btnCompare" class="btn-weg-action w-100 mt-1">⚡ Executar Metrologia Óptica Comparativa</button>
                    </div>
                </div>

                <div class="col-lg-7">
                    <div class="weg-card">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5 class="fw-bold mb-0" style="color: var(--weg-cyan);">2. Laudo Metrológico Automatizado</h5>
                            <span id="badgeStatusVision" class="badge bg-secondary px-3 py-2">Aguardando Execução</span>
                        </div>

                        <div id="visionEmptyState" class="text-center py-5 my-4" style="color: var(--text-muted);">
                            <p class="mb-1">Carregue as imagens de referência (`samples/geometry/motor_reference.png`) e de teste (`motor_bad.png` ou `motor_rust.png`).</p>
                            <small>O algoritmo extrairá os contornos via Canny Edge Detection e destacará as avarias em vermelho néon.</small>
                        </div>

                        <div id="visionResult" style="display: none;">
                            <div class="row g-2 mb-3">
                                <div class="col-md-3">
                                    <div class="kpi-box py-2">
                                        <small class="kpi-label d-block">Circularidade Ref.</small>
                                        <span id="metricCircRef" class="fs-5 fw-bold text-white">---</span>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="kpi-box py-2">
                                        <small class="kpi-label d-block">Circularidade Teste</small>
                                        <span id="metricCircTest" class="fs-5 fw-bold text-white">---</span>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="kpi-box py-2">
                                        <small class="kpi-label d-block">Δ Circularidade</small>
                                        <span id="metricDeltaCirc" class="fs-5 fw-bold text-white">---</span>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="kpi-box py-2" style="border-top-color: #ef4444;">
                                        <small class="kpi-label d-block">Desgaste Calculado</small>
                                        <span id="metricWear" class="fs-5 fw-bold text-danger">---</span>
                                    </div>
                                </div>
                            </div>

                            <div class="row g-3 text-center">
                                <div class="col-md-4">
                                    <small class="d-block mb-1 fw-bold" style="color: var(--weg-cyan);">Bordas Canny (Padrão)</small>
                                    <div class="preview-box"><img id="imgCannyRef" src=""></div>
                                </div>
                                <div class="col-md-4">
                                    <small class="d-block mb-1 fw-bold text-warning">Bordas Canny (Teste)</small>
                                    <div class="preview-box"><img id="imgCannyTest" src=""></div>
                                </div>
                                <div class="col-md-4">
                                    <small class="d-block mb-1 fw-bold text-danger">Mapa de Avarias (Vermelho)</small>
                                    <div class="preview-box"><img id="imgOverlayDesgaste" src=""></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- SEÇÃO 3: INSPEÇÃO TERMOGRÁFICA INFRAVERMELHA NO ESPAÇO HSV -->
        <section id="sec-thermal" class="section-block">
            <span class="section-header-tag">ETAPA 3.2 • VISÃO COMPUTACIONAL TERMOMECÂNICA</span>
            <h2 class="section-title">3. Inspeção Termográfica Infravermelha (Detecção de Hotspots HSV)</h2>

            <div class="row g-4">
                <div class="col-lg-5">
                    <div class="weg-card">
                        <h5 class="fw-bold mb-3" style="color: var(--weg-cyan);">1. Frame da Câmera Infravermelha</h5>
                        <p class="small" style="color: var(--text-muted);">
                            Converte o mapa de calor para o espaço cromático <strong>HSV (Hue, Saturation, Value)</strong> e aplica limiarização para isolar zonas de superaquecimento crítico nos mancais e rotores.
                        </p>
                        <label class="form-label small fw-bold">Selecione o frame térmico (`samples/thermal/`):</label>
                        <input type="file" id="fileThermal" class="form-control form-control-sm mb-3" accept="image/*">
                        <div class="preview-box mb-3" id="boxThermalPreview"><span class="text-secondary small">Preview da imagem térmica...</span></div>
                        <button id="btnThermal" class="btn-weg-action w-100">🔥 Segmentar Hotspots Críticos (Espaço HSV)</button>
                    </div>
                </div>

                <div class="col-lg-7">
                    <div class="weg-card">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5 class="fw-bold mb-0" style="color: var(--weg-cyan);">2. Diagnóstico Termográfico</h5>
                            <span id="badgeThermal" class="badge bg-secondary px-3 py-2">Aguardando Frame</span>
                        </div>

                        <div id="thermalEmpty" class="text-center py-5 my-4" style="color: var(--text-muted);">
                            <p class="mb-1">Envie `termografia_normal.png` ou `termografia_hotspot_critico.png` para validar a segmentação HSV.</p>
                        </div>

                        <div id="thermalResult" style="display: none;">
                            <div class="kpi-box mb-3 text-start d-flex justify-content-between align-items-center px-4">
                                <div>
                                    <span class="kpi-label d-block">Área Superficial em Regime Crítico de Temperatura</span>
                                    <small class="text-secondary">Limiar de segurança configurado na matriz HSV</small>
                                </div>
                                <span id="thermalRatio" class="fs-3 fw-bold text-danger">0.00%</span>
                            </div>

                            <div class="row g-3 text-center">
                                <div class="col-md-6">
                                    <small class="d-block mb-1 fw-bold" style="color: var(--weg-cyan);">Máscara Binária Isolada (Hotspot HSV)</small>
                                    <div class="preview-box" style="height: 240px;"><img id="imgThermalMask" src=""></div>
                                </div>
                                <div class="col-md-6">
                                    <small class="d-block mb-1 fw-bold text-warning">Sobreposição de Zonas Térmicas Críticas</small>
                                    <div class="preview-box" style="height: 240px;"><img id="imgThermalOverlay" src=""></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- SEÇÃO 4: SIMULADOR DE TELEMETRIA IOT & DIAGNÓSTICO PREDITIVO -->
        <section id="sec-telemetry" class="section-block border-bottom-0">
            <span class="section-header-tag">ETAPA 3.3 • INTEGRAÇÃO CLP / SCADA EM TEMPO REAL</span>
            <h2 class="section-title">4. Simulador de Telemetria IoT & Inferência Preditiva Multiclasse</h2>

            <div class="row g-4">
                <div class="col-lg-6">
                    <div class="weg-card">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5 class="fw-bold mb-0" style="color: var(--weg-cyan);">1. Parâmetros Instantâneos dos Sensores</h5>
                            <div class="btn-group btn-group-sm">
                                <button type="button" class="btn btn-outline-info" onclick="carregarPreset('normal')">Preset Nominal</button>
                                <button type="button" class="btn btn-outline-warning" onclick="carregarPreset('desbalanceamento')">Preset Desbalanceamento</button>
                                <button type="button" class="btn btn-outline-danger" onclick="carregarPreset('cavitacao')">Preset Cavitação</button>
                            </div>
                        </div>

                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label small fw-bold">Bancada de Teste:</label>
                                <select id="telEq" class="form-select">
                                    <option value="TURB-101">TURB-101 (Bancada Principal)</option>
                                    <option value="TURB-102">TURB-102 (Alta Rotação)</option>
                                    <option value="TURB-103" selected>TURB-103 (Ensaio de Carga)</option>
                                    <option value="TURB-104">TURB-104 (Validação Térmica)</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-bold">Regime Operacional:</label>
                                <select id="telModo" class="form-select">
                                    <option value="Cruzeiro">Cruzeiro</option>
                                    <option value="Carga_Media">Carga_Media</option>
                                    <option value="Carga_Alta" selected>Carga_Alta</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-bold">Vibração RMS (mm/s):</label>
                                <input type="number" id="telVib" class="form-control" value="9.8" step="0.1">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-bold">Temp. Mancal (°C):</label>
                                <input type="number" id="telTemp" class="form-control" value="72.0" step="0.5">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-bold">Rotação (RPM):</label>
                                <input type="number" id="telRpm" class="form-control" value="3450" step="10">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-bold">Pressão Óleo (Bar):</label>
                                <input type="number" id="telPressao" class="form-control" value="4.3" step="0.1">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-bold">Emissão Acústica (dB):</label>
                                <input type="number" id="telAcustica" class="form-control" value="55.0" step="1.0">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-bold">Corrente (A):</label>
                                <input type="number" id="telCorrente" class="form-control" value="38.0" step="0.5">
                            </div>
                        </div>

                        <button id="btnTelPredict" class="btn-weg-action w-100 mt-4">📡 Executar Inferência Preditiva (XGBoost + Isolation Forest)</button>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="weg-card">
                        <h5 class="fw-bold mb-3" style="color: var(--weg-cyan);">2. Diagnóstico Prescritivo & Probabilidades</h5>
                        <div id="telResult" class="py-2">
                            <div class="kpi-box text-start px-4 py-3 mb-3">
                                <small class="kpi-label d-block">Estado Operacional Predito pela IA:</small>
                                <h3 id="telDiag" class="fw-bold text-white mb-0">Clique em Executar Inferência...</h3>
                            </div>
                            <div id="telProbList" class="mt-3"></div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

    </main>

    <!-- RODAPÉ ESTILO WEG -->
    <footer class="text-center">
        <div class="container">
            <p class="mb-1 fw-bold text-white">AeroMotion Dynamics S.A. • Plataforma Expert Industrial AI</p>
            <small>SENAI | CentroWEG • Curso de Cibersistemas para Automação • Jaraguá do Sul, SC — 2026 • Desenvolvido por Henrique Schorck</small>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        Chart.defaults.color = '#94b2d1';
        Chart.defaults.borderColor = 'rgba(25, 75, 133, 0.35)';

        async function initAnalyticsCharts() {
            try {
                const res = await fetch('/analytics/summary');
                const data = await res.json();

                new Chart(document.getElementById('chartFaultSignatures'), {
                    type: 'bar',
                    data: {
                        labels: data.assinatura_fisica_falhas.classes,
                        datasets: [
                            {
                                label: 'Temperatura Mancal (°C)',
                                data: data.assinatura_fisica_falhas.temperatura_c,
                                backgroundColor: '#00579d'
                            },
                            {
                                label: 'Emissão Acústica (dB)',
                                data: data.assinatura_fisica_falhas.emissao_acustica_db,
                                backgroundColor: '#5bc2e7'
                            },
                            {
                                label: 'Vibração RMS (mm/s × 5)',
                                data: data.assinatura_fisica_falhas.vibracao_mms.map(v => v * 5),
                                backgroundColor: '#f59e0b'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { position: 'top' } }
                    }
                });

                new Chart(document.getElementById('chartFeatureImportance'), {
                    type: 'bar',
                    data: {
                        labels: data.importancia_features.features,
                        datasets: [{
                            label: 'Importância Relativa (%)',
                            data: data.importancia_features.pesos,
                            backgroundColor: '#5bc2e7',
                            borderColor: '#00579d',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } }
                    }
                });

                new Chart(document.getElementById('chartOperatingRegimes'), {
                    type: 'bar',
                    data: {
                        labels: data.regimes_operacao.regimes,
                        datasets: [
                            {
                                label: 'Temperatura Média (°C)',
                                data: data.regimes_operacao.temp_media,
                                backgroundColor: '#00579d'
                            },
                            {
                                label: 'Vibração Média (mm/s × 10)',
                                data: data.regimes_operacao.vib_media.map(v => v * 10),
                                backgroundColor: '#38bdf8'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });

                new Chart(document.getElementById('chartClassDistribution'), {
                    type: 'doughnut',
                    data: {
                        labels: data.distribuicao_classes.labels,
                        datasets: [{
                            data: data.distribuicao_classes.counts,
                            backgroundColor: ['#00579d', '#5bc2e7', '#f59e0b', '#ef4444'],
                            borderColor: '#072244',
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { position: 'right' } }
                    }
                });
            } catch (e) {
                console.error('Erro ao inicializar gráficos:', e);
            }
        }
        initAnalyticsCharts();

        async function monitorarEdgeNode() {
            const badge = document.getElementById('badgeEdgeStatus');
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 2000);
                const res = await fetch('/health', { signal: controller.signal });
                clearTimeout(timeoutId);
                if (!res.ok) throw new Error();
                const data = await res.json();
                if (data.status === 'online') {
                    badge.className = 'badge bg-success badge-heartbeat';
                    badge.innerHTML = '<span class="status-led"></span> Edge Node Ativo';
                } else {
                    badge.className = 'badge bg-warning text-dark badge-heartbeat';
                    badge.innerHTML = '<span class="status-led" style="background:#000;"></span> Edge Node Degradado';
                }
            } catch (err) {
                badge.className = 'badge bg-danger badge-heartbeat';
                badge.innerHTML = '<span class="status-led"></span> Edge Node Desconectado';
            }
        }
        monitorarEdgeNode();
        setInterval(monitorarEdgeNode, 3000);

        document.getElementById('rngWear').addEventListener('input', (e) => document.getElementById('lblWear').innerText = e.target.value + '%');
        document.getElementById('rngCirc').addEventListener('input', (e) => document.getElementById('lblCirc').innerText = e.target.value);
        document.getElementById('rngDiff').addEventListener('input', (e) => document.getElementById('lblDiff').innerText = e.target.value);
        document.getElementById('rngCanny').addEventListener('input', (e) => document.getElementById('lblCanny').innerText = e.target.value);

        function bindPreview(inputId, boxId) {
            document.getElementById(inputId).addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = (ev) => {
                        document.getElementById(boxId).innerHTML = `<img src="${ev.target.result}">`;
                    };
                    reader.readAsDataURL(file);
                }
            });
        }
        bindPreview('fileRef', 'boxRef');
        bindPreview('fileTest', 'boxTest');
        bindPreview('fileThermal', 'boxThermalPreview');

        document.getElementById('btnCompare').addEventListener('click', async () => {
            const fRef = document.getElementById('fileRef').files[0];
            const fTest = document.getElementById('fileTest').files[0];
            if (!fRef || !fTest) return alert('Selecione a imagem de referência e a imagem em teste!');

            const fd = new FormData();
            fd.append('file_ref', fRef);
            fd.append('file_test', fTest);
            fd.append('max_desgaste_pct', document.getElementById('rngWear').value);
            fd.append('min_circularidade', document.getElementById('rngCirc').value);
            fd.append('diff_threshold', document.getElementById('rngDiff').value);
            fd.append('canny_high', document.getElementById('rngCanny').value);

            const btn = document.getElementById('btnCompare');
            btn.disabled = true;
            btn.innerText = 'Processando Visão Computacional...';

            try {
                const res = await fetch('/inspect/compare', { method: 'POST', body: fd });
                const data = await res.json();
                document.getElementById('visionEmptyState').style.display = 'none';
                document.getElementById('visionResult').style.display = 'block';

                const r = data.relatorio;
                const badge = document.getElementById('badgeStatusVision');
                badge.innerText = r.status;
                badge.className = 'badge px-3 py-2 ' + (r.aprovado ? 'bg-success' : 'bg-danger');

                document.getElementById('metricCircRef').innerText = r.circularidade_referencia;
                document.getElementById('metricCircTest').innerText = r.circularidade_teste;
                document.getElementById('metricDeltaCirc').innerText = r.delta_circularidade;
                document.getElementById('metricWear').innerText = r.desgaste_superficie_pct + '%';

                document.getElementById('imgCannyRef').src = r.canny_ref_b64;
                document.getElementById('imgCannyTest').src = r.canny_test_b64;
                document.getElementById('imgOverlayDesgaste').src = r.overlay_desgaste_b64;
            } catch (err) {
                alert('Erro na comunicação com o servidor.');
            } finally {
                btn.disabled = false;
                btn.innerText = '⚡ Executar Metrologia Óptica Comparativa';
            }
        });

        document.getElementById('btnThermal').addEventListener('click', async () => {
            const f = document.getElementById('fileThermal').files[0];
            if (!f) return alert('Selecione um frame infravermelho!');
            const fd = new FormData();
            fd.append('file', f);
            const res = await fetch('/inspect/thermal', { method: 'POST', body: fd });
            const data = await res.json();

            document.getElementById('thermalEmpty').style.display = 'none';
            document.getElementById('thermalResult').style.display = 'block';

            const m = data.metricas;
            const b = document.getElementById('badgeThermal');
            b.innerText = m.status;
            b.className = 'badge px-3 py-2 ' + (m.alarme_termico ? 'bg-danger' : 'bg-success');
            document.getElementById('thermalRatio').innerText = m.taxa_area_critica_pct + '%';
            document.getElementById('imgThermalMask').src = m.mask_b64;
            document.getElementById('imgThermalOverlay').src = m.overlay_b64;
        });

        function carregarPreset(tipo) {
            if (tipo === 'normal') {
                document.getElementById('telModo').value = 'Cruzeiro';
                document.getElementById('telVib').value = 2.1;
                document.getElementById('telTemp').value = 58.0;
                document.getElementById('telRpm').value = 2450;
                document.getElementById('telPressao').value = 4.3;
                document.getElementById('telAcustica').value = 49.0;
                document.getElementById('telCorrente').value = 24.5;
            } else if (tipo === 'desbalanceamento') {
                document.getElementById('telModo').value = 'Carga_Alta';
                document.getElementById('telVib').value = 10.4;
                document.getElementById('telTemp').value = 65.0;
                document.getElementById('telRpm').value = 3480;
                document.getElementById('telPressao').value = 4.2;
                document.getElementById('telAcustica').value = 54.0;
                document.getElementById('telCorrente').value = 39.0;
            } else if (tipo === 'cavitacao') {
                document.getElementById('telModo').value = 'Carga_Alta';
                document.getElementById('telVib').value = 3.1;
                document.getElementById('telTemp').value = 98.5;
                document.getElementById('telRpm').value = 3510;
                document.getElementById('telPressao').value = 1.9;
                document.getElementById('telAcustica').value = 56.0;
                document.getElementById('telCorrente').value = 41.2;
            }
        }

        document.getElementById('btnTelPredict').addEventListener('click', async () => {
            const payload = {
                id_equipamento: document.getElementById('telEq').value,
                modo_operacao: document.getElementById('telModo').value,
                horas_operacao_acumuladas: 1500.0,
                temperatura_mancal_c: parseFloat(document.getElementById('telTemp').value),
                vibracao_rms_mms: parseFloat(document.getElementById('telVib').value),
                pressao_oleo_bar: parseFloat(document.getElementById('telPressao').value),
                rotacao_rpm: parseFloat(document.getElementById('telRpm').value),
                emissao_acustica_db: parseFloat(document.getElementById('telAcustica').value),
                corrente_eletrica_amp: parseFloat(document.getElementById('telCorrente').value)
            };

            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            document.getElementById('telDiag').innerText = data.diagnostico_descricao + ' (Classe ' + data.diagnostico_codigo + ')';

            let html = '<h6 class="text-white small fw-bold text-uppercase mt-2 mb-3">Distribuição de Probabilidades (XGBoost):</h6>';
            for (const [k, v] of Object.entries(data.probabilidades)) {
                const pct = (v * 100).toFixed(1);
                const barColor = (k === 'Normal') ? '#22c55e' : '#5bc2e7';
                html += `
                <div class="mb-2">
                    <div class="d-flex justify-content-between small mb-1">
                        <span>${k}</span>
                        <span class="fw-bold text-white">${pct}%</span>
                    </div>
                    <div class="progress" style="height: 8px; background: #031124;">
                        <div class="progress-bar" style="width: ${pct}%; background-color: ${barColor};"></div>
                    </div>
                </div>`;
            }
            if (data.alerta_anomalia_desconhecida) {
                html += '<div class="alert alert-warning mt-3 py-2 small mb-0">⚠️ <strong>Sentinela Isolation Forest:</strong> Comportamento multivariado anômalo detectado!</div>';
            }
            document.getElementById('telProbList').innerHTML = html;
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def serve_dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(BASE_DIR))