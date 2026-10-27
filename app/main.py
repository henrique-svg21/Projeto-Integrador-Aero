"""
Microsserviço RESTful e Dashboard Web Industrial com Sliders de Tolerância e Heartbeat.
AeroMotion Dynamics S.A.
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
        print("-> Modelos de IA carregados com sucesso.")
    else:
        print("-> AVISO: Modelos nao encontrados em models/.")
    yield
    artifacts.clear()


app = FastAPI(
    title="AeroMotion Dynamics - Industrial Control",
    version="1.3.1",
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
        "model_version": "1.3.1-heartbeat",
        "model_loaded": model_loaded
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
    <title>AeroMotion Dynamics - Central de Metrologia & Visão</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #0b1329; color: #f1f5f9; font-family: 'Segoe UI', system-ui, sans-serif; }
        .card { background-color: #16203a; border: 1px solid #283759; border-radius: 12px; }
        .nav-tabs .nav-link { color: #94a3b8; border: none; font-weight: 600; }
        .nav-tabs .nav-link.active { color: #38bdf8; background-color: #16203a; border-bottom: 3px solid #38bdf8; }
        .preview-box { height: 210px; border: 2px dashed #334155; border-radius: 8px; display: flex; align-items: center; justify-content: center; overflow: hidden; background: #080d1a; }
        .preview-box img { max-height: 100%; max-width: 100%; object-fit: contain; }
        .badge-status { font-size: 1.05rem; padding: 8px 16px; border-radius: 8px; transition: all 0.3s ease; }
        .metric-card { background: #0c1527; border-radius: 8px; padding: 12px; border-left: 4px solid #38bdf8; }
        .slider-label { font-size: 0.88rem; font-weight: 600; color: #cbd5e1; display: flex; justify-content: space-between; margin-bottom: 4px; }
        .form-range::-webkit-slider-thumb { background: #38bdf8; }
    </style>
</head>
<body class="p-4">
    <div class="container-fluid max-w-7xl">
        <!-- Header -->
        <div class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom border-secondary">
            <div>
                <h3 class="fw-bold text-sky-400 mb-0">✈️ AeroMotion Dynamics S.A.</h3>
                <small class="text-secondary">Controle de Qualidade Óptica & Metrologia Parametrizável de Componentes</small>
            </div>
            <div class="d-flex align-items-center">
                <!-- Badge Dinâmico com Monitoramento de Conexão -->
                <span id="badgeEdgeStatus" class="badge bg-secondary badge-status me-2">⌛ Verificando nó...</span>
                <a href="/docs" target="_blank" class="btn btn-outline-info btn-sm">Documentação Swagger API</a>
            </div>
        </div>

        <!-- Abas -->
        <ul class="nav nav-tabs mb-4" id="mainTab" role="tablist">
            <li class="nav-item">
                <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#vision-pane">🔍 Inspeção Visual & Tolerâncias</button>
            </li>
            <li class="nav-item">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#thermal-pane">🔥 Termografia HSV</button>
            </li>
            <li class="nav-item">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#telemetry-pane">📊 Telemetria Preditiva IoT</button>
            </li>
        </ul>

        <div class="tab-content">
            <!-- ABA 1: VISÃO COM SLIDERS -->
            <div class="tab-pane fade show active" id="vision-pane">
                <div class="row g-4">
                    <div class="col-lg-5">
                        <div class="card p-3 shadow mb-3">
                            <h5 class="card-title text-info mb-3">1. Imagens para Comparação</h5>
                            
                            <label class="form-label small fw-bold">Imagem de Referência (Nominal / Boa):</label>
                            <input type="file" id="fileRef" class="form-control form-control-sm mb-2 bg-dark text-light border-secondary" accept="image/*">
                            <div class="preview-box mb-3" id="boxRef"><span class="text-muted small">Aguardando referência...</span></div>

                            <label class="form-label small fw-bold">Imagem em Teste (Com Possível Desgaste):</label>
                            <input type="file" id="fileTest" class="form-control form-control-sm mb-2 bg-dark text-light border-secondary" accept="image/*">
                            <div class="preview-box" id="boxTest"><span class="text-muted small">Aguardando peça sob teste...</span></div>
                        </div>

                        <div class="card p-3 shadow">
                            <h5 class="card-title text-warning mb-3">⚙️ Tolerâncias Industriais & Filtros</h5>

                            <div class="mb-3">
                                <div class="slider-label">
                                    <span>Tolerância Máx. Desgaste Superficial:</span>
                                    <span id="lblWear" class="text-info fw-bold">10%</span>
                                </div>
                                <input type="range" class="form-range" id="rngWear" min="1" max="60" step="1" value="10">
                                <small class="text-secondary d-block" style="font-size: 0.75rem;">Peças críticas: 2-5% | Peças de carcaça: 15-40%</small>
                            </div>

                            <div class="mb-3">
                                <div class="slider-label">
                                    <span>Circularidade Mínima de Aceitação:</span>
                                    <span id="lblCirc" class="text-info fw-bold">0.70</span>
                                </div>
                                <input type="range" class="form-range" id="rngCirc" min="0.40" max="0.95" step="0.02" value="0.70">
                                <small class="text-secondary d-block" style="font-size: 0.75rem;">1.0 = Círculo Perfeito. Reprova peças ovalizadas ou entalhadas.</small>
                            </div>

                            <div class="mb-3">
                                <div class="slider-label">
                                    <span>Sensibilidade de Diferença (Filtro de Luz):</span>
                                    <span id="lblDiff" class="text-info fw-bold">35</span>
                                </div>
                                <input type="range" class="form-range" id="rngDiff" min="15" max="90" step="1" value="35">
                                <small class="text-secondary d-block" style="font-size: 0.75rem;">Aumente para ignorar sombras e diferenças leves de tom de pintura.</small>
                            </div>

                            <div class="mb-3">
                                <div class="slider-label">
                                    <span>Limiar Superior de Borda (Canny High):</span>
                                    <span id="lblCanny" class="text-info fw-bold">150</span>
                                </div>
                                <input type="range" class="form-range" id="rngCanny" min="80" max="250" step="5" value="150">
                                <small class="text-secondary d-block" style="font-size: 0.75rem;">Controla a sensibilidade de bordas finas vs contorno geral da peça.</small>
                            </div>

                            <button id="btnCompare" class="btn btn-primary w-100 fw-bold py-2 mt-2">⚡ Executar Análise com Tolerâncias</button>
                        </div>
                    </div>

                    <div class="col-lg-7">
                        <div class="card p-4 shadow h-100">
                            <h5 class="card-title text-info mb-3">2. Diagnóstico Metrológico em Tempo Real</h5>
                            
                            <div id="visionEmptyState" class="text-center py-5 text-muted">
                                <h6>Selecione as imagens e clique no botão para inspecionar com as tolerâncias configuradas.</h6>
                            </div>

                            <div id="visionResult" style="display: none;">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <h4 class="mb-0">Parecer Final:</h4>
                                    <span id="badgeStatusVision" class="badge">---</span>
                                </div>

                                <div class="row g-2 mb-4">
                                    <div class="col-md-3">
                                        <div class="metric-card">
                                            <small class="text-secondary">Circularidade Ref.</small>
                                            <h5 id="metricCircRef" class="mb-0 text-white">---</h5>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="metric-card">
                                            <small class="text-secondary">Circularidade Teste</small>
                                            <h5 id="metricCircTest" class="mb-0 text-white">---</h5>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="metric-card">
                                            <small class="text-secondary">Δ Circularidade</small>
                                            <h5 id="metricDeltaCirc" class="mb-0 text-white">---</h5>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="metric-card">
                                            <small class="text-secondary">Desgaste Calculado</small>
                                            <h5 id="metricWear" class="mb-0 text-danger">---</h5>
                                        </div>
                                    </div>
                                </div>

                                <h6 class="text-secondary border-bottom pb-2">Evidências Visuais e Extração de Contornos (Canny):</h6>
                                <div class="row g-3 text-center">
                                    <div class="col-6">
                                        <small class="d-block mb-1 text-info">Bordas Canny (Referência)</small>
                                        <div class="preview-box"><img id="imgCannyRef" src=""></div>
                                    </div>
                                    <div class="col-6">
                                        <small class="d-block mb-1 text-warning">Bordas Canny (Peça em Teste)</small>
                                        <div class="preview-box"><img id="imgCannyTest" src=""></div>
                                    </div>
                                    <div class="col-12 mt-3">
                                        <small class="d-block mb-1 text-danger fw-bold">Zonas de Avaria / Desgaste Detectadas (Destaque em Vermelho):</small>
                                        <div class="preview-box" style="height: 310px;"><img id="imgOverlayDesgaste" src=""></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ABA 2: TERMOGRAFIA HSV -->
            <div class="tab-pane fade" id="thermal-pane">
                <div class="row g-4">
                    <div class="col-lg-5">
                        <div class="card p-3 shadow">
                            <h5 class="card-title text-warning mb-3">Termografia Infravermelha</h5>
                            <label class="form-label text-light">Selecione o frame infravermelho da turbina:</label>
                            <input type="file" id="fileThermal" class="form-control mb-3 bg-dark text-light border-secondary" accept="image/*">
                            <button id="btnThermal" class="btn btn-warning w-100 fw-bold py-2">🔥 Delimitar Hotspots (Espaço HSV)</button>
                        </div>
                    </div>
                    <div class="col-lg-7">
                        <div class="card p-4 shadow">
                            <h5 class="card-title text-warning mb-3">Análise de Zonas Críticas</h5>
                            <div id="thermalResult" style="display: none;">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <h4>Alarme Térmico:</h4>
                                    <span id="badgeThermal" class="badge">---</span>
                                </div>
                                <p class="mb-3">Taxa de Superaquecimento Crítico: <strong id="thermalRatio" class="text-danger">---</strong></p>
                                <div class="row g-3 text-center">
                                    <div class="col-6">
                                        <small class="d-block mb-1 text-secondary">Máscara Binária de Hotspot</small>
                                        <div class="preview-box"><img id="imgThermalMask" src=""></div>
                                    </div>
                                    <div class="col-6">
                                        <small class="d-block mb-1 text-secondary">Sobreposição Térmica</small>
                                        <div class="preview-box"><img id="imgThermalOverlay" src=""></div>
                                    </div>
                                </div>
                            </div>
                            <div id="thermalEmpty" class="text-center py-5 text-muted">Carregue um frame térmico para processamento.</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ABA 3: TELEMETRIA -->
            <div class="tab-pane fade" id="telemetry-pane">
                <div class="row g-4">
                    <div class="col-lg-6">
                        <div class="card p-4 shadow">
                            <h5 class="card-title text-info mb-3">Simulador de Telemetria de Bancada</h5>
                            <div class="row g-2">
                                <div class="col-6">
                                    <label class="form-label text-secondary small">ID Bancada:</label>
                                    <select id="telEq" class="form-select bg-dark text-light border-secondary">
                                        <option value="TURB-101">TURB-101</option>
                                        <option value="TURB-102">TURB-102</option>
                                        <option value="TURB-103" selected>TURB-103</option>
                                        <option value="TURB-104">TURB-104</option>
                                    </select>
                                </div>
                                <div class="col-6">
                                    <label class="form-label text-secondary small">Regime de Carga:</label>
                                    <select id="telModo" class="form-select bg-dark text-light border-secondary">
                                        <option value="Cruzeiro">Cruzeiro</option>
                                        <option value="Carga_Media">Carga_Media</option>
                                        <option value="Carga_Alta" selected>Carga_Alta</option>
                                    </select>
                                </div>
                                <div class="col-6 mt-2">
                                    <label class="form-label text-secondary small">Vibração RMS (mm/s):</label>
                                    <input type="number" id="telVib" class="form-control bg-dark text-light border-secondary" value="9.8" step="0.1">
                                </div>
                                <div class="col-6 mt-2">
                                    <label class="form-label text-secondary small">Temperatura Mancal (°C):</label>
                                    <input type="number" id="telTemp" class="form-control bg-dark text-light border-secondary" value="78.5" step="0.5">
                                </div>
                                <div class="col-6 mt-2">
                                    <label class="form-label text-secondary small">Rotação (RPM):</label>
                                    <input type="number" id="telRpm" class="form-control bg-dark text-light border-secondary" value="3450">
                                </div>
                                <div class="col-6 mt-2">
                                    <label class="form-label text-secondary small">Pressão Óleo (Bar):</label>
                                    <input type="number" id="telPressao" class="form-control bg-dark text-light border-secondary" value="4.3" step="0.1">
                                </div>
                            </div>
                            <button id="btnTelPredict" class="btn btn-info w-100 fw-bold mt-4">Diagnosticar Telemetria (XGBoost)</button>
                        </div>
                    </div>
                    <div class="col-lg-6">
                        <div class="card p-4 shadow h-100">
                            <h5 class="card-title text-info mb-3">Diagnóstico Preditivo em Tempo Real</h5>
                            <div id="telResult" class="py-3">
                                <h4 id="telDiag" class="text-white">Aguardando inferência...</h4>
                                <div id="telProbList" class="mt-3"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Sincronização em tempo real dos valores dos sliders
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

        // Monitoramento Ativo do Edge Node (Heartbeat contínuo)
        async function monitorarEdgeNode() {
            const badge = document.getElementById('badgeEdgeStatus');
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 2000); // Timeout de 2s
                const res = await fetch('/health', { signal: controller.signal });
                clearTimeout(timeoutId);

                if (!res.ok) throw new Error('Status inesperado');
                const data = await res.json();

                if (data.status === 'online') {
                    badge.className = 'badge bg-success badge-status me-2';
                    badge.innerHTML = '● Edge Node Ativo';
                } else {
                    badge.className = 'badge bg-warning text-dark badge-status me-2';
                    badge.innerHTML = '▲ Edge Node Degradado (Sem Modelos)';
                }
            } catch (err) {
                badge.className = 'badge bg-danger badge-status me-2';
                badge.innerHTML = '✖ Edge Node Desconectado';
            }
        }

        // Executa imediatamente e checa a cada 3 segundos
        monitorarEdgeNode();
        setInterval(monitorarEdgeNode, 3000);

        // Comparação com envio de tolerâncias
        document.getElementById('btnCompare').addEventListener('click', async () => {
            const fRef = document.getElementById('fileRef').files[0];
            const fTest = document.getElementById('fileTest').files[0];
            if (!fRef || !fTest) {
                alert('Por favor, selecione tanto a peça de referência quanto a peça em teste!');
                return;
            }

            const formData = new FormData();
            formData.append('file_ref', fRef);
            formData.append('file_test', fTest);
            formData.append('max_desgaste_pct', document.getElementById('rngWear').value);
            formData.append('min_circularidade', document.getElementById('rngCirc').value);
            formData.append('diff_threshold', document.getElementById('rngDiff').value);
            formData.append('canny_high', document.getElementById('rngCanny').value);

            const btn = document.getElementById('btnCompare');
            btn.disabled = true;
            btn.innerText = 'Processando com Tolerâncias...';

            try {
                const res = await fetch('/inspect/compare', { method: 'POST', body: formData });
                const data = await res.json();
                
                document.getElementById('visionEmptyState').style.display = 'none';
                document.getElementById('visionResult').style.display = 'block';

                const r = data.relatorio;
                const badge = document.getElementById('badgeStatusVision');
                badge.innerText = r.status;
                badge.className = 'badge ' + (r.aprovado ? 'bg-success' : 'bg-danger');

                document.getElementById('metricCircRef').innerText = r.circularidade_referencia;
                document.getElementById('metricCircTest').innerText = r.circularidade_teste;
                document.getElementById('metricDeltaCirc').innerText = r.delta_circularidade;
                document.getElementById('metricWear').innerText = r.desgaste_superficie_pct + '%';

                document.getElementById('imgCannyRef').src = r.canny_ref_b64;
                document.getElementById('imgCannyTest').src = r.canny_test_b64;
                document.getElementById('imgOverlayDesgaste').src = r.overlay_desgaste_b64;
            } catch (err) {
                alert('Erro na comunicação com a API.');
            } finally {
                btn.disabled = false;
                btn.innerText = '⚡ Executar Análise com Tolerâncias';
            }
        });

        // Termografia
        document.getElementById('btnThermal').addEventListener('click', async () => {
            const f = document.getElementById('fileThermal').files[0];
            if (!f) return alert('Selecione uma imagem térmica!');
            const fd = new FormData();
            fd.append('file', f);
            const res = await fetch('/inspect/thermal', { method: 'POST', body: fd });
            const data = await res.json();
            
            document.getElementById('thermalEmpty').style.display = 'none';
            document.getElementById('thermalResult').style.display = 'block';

            const m = data.metricas;
            const b = document.getElementById('badgeThermal');
            b.innerText = m.status;
            b.className = 'badge ' + (m.alarme_termico ? 'bg-danger' : 'bg-success');
            document.getElementById('thermalRatio').innerText = m.taxa_area_critica_pct + '% da área total';
            document.getElementById('imgThermalMask').src = m.mask_b64;
            document.getElementById('imgThermalOverlay').src = m.overlay_b64;
        });

        // Telemetria Predict
        document.getElementById('btnTelPredict').addEventListener('click', async () => {
            const payload = {
                id_equipamento: document.getElementById('telEq').value,
                modo_operacao: document.getElementById('telModo').value,
                horas_operacao_acumuladas: 1500.0,
                temperatura_mancal_c: parseFloat(document.getElementById('telTemp').value),
                vibracao_rms_mms: parseFloat(document.getElementById('telVib').value),
                pressao_oleo_bar: parseFloat(document.getElementById('telPressao').value),
                rotacao_rpm: parseFloat(document.getElementById('telRpm').value),
                emissao_acustica_db: 55.0,
                corrente_eletrica_amp: 35.0
            };
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            document.getElementById('telDiag').innerText = 'Diagnóstico: ' + data.diagnostico_descricao;
            
            let html = '<h6 class="text-secondary mt-3">Probabilidades Calculadas:</h6>';
            for (const [k, v] of Object.entries(data.probabilidades)) {
                const pct = (v * 100).toFixed(1);
                html += `<div class="mb-2"><small>${k}: ${pct}%</small><div class="progress" style="height: 6px;"><div class="progress-bar bg-info" style="width: ${pct}%"></div></div></div>`;
            }
            if (data.alerta_anomalia_desconhecida) {
                html += '<div class="alert alert-warning mt-3 py-2 small">⚠️ Alerta Isolation Forest: Anomalia não documentada detectada!</div>';
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