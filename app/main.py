"""
Microsserviço RESTful de Inferência Preditiva em Tempo Real - AeroMotion Dynamics.
Compatível com FastAPI moderno (Lifespan), Pydantic V2 e execução em Windows (Python 3.12).
"""
from contextlib import asynccontextmanager
from enum import Enum
import os
from pathlib import Path
import sys
from typing import Any, Dict, List
import joblib
import pandas as pd
from pydantic import BaseModel, Field
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse

# Garante que a raiz do projeto esteja no sys.path em qualquer ambiente
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Caminhos absolutos e robustos para os modelos
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
    """Gerenciador moderno de ciclo de vida (Lifespan) do FastAPI."""
    if MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
        artifacts["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
        artifacts["classifier"] = joblib.load(MODEL_PATH)
        if ISO_PATH.exists():
            artifacts["anomaly_detector"] = joblib.load(ISO_PATH)
        print("-> Artefatos preditivos carregados com sucesso em memoria.")
    else:
        print("-> AVISO: Modelos nao encontrados em models/. Execute src/train_model.py.")
    yield
    artifacts.clear()


app = FastAPI(
    title="AeroMotion Dynamics - Telemetry Predictive API",
    version="1.0.0",
    description="Microservico industrial para diagnostico precoce de falhas em turbinas.",
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


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    model_loaded = "classifier" in artifacts
    return {
        "status": "online" if model_loaded else "degraded",
        "service": "AeroMotion Predictive Inference API",
        "model_version": "1.0.0-xgb-rf",
        "model_loaded": model_loaded
    }


@app.post("/predict", response_model=PredicaoResposta)
def predict_single(telemetria: TelemetriaLeitura):
    if "classifier" not in artifacts or "preprocessor" not in artifacts:
        raise HTTPException(status_code=503, detail="Modelos nao carregados no servidor.")

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
        raise HTTPException(status_code=503, detail="Modelos nao carregados no servidor.")

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


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(BASE_DIR))