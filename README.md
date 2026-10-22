<div align="center">

# AeroMotion Dynamics S.A.
### Plataforma Integrada de Inteligência Artificial & Visão Computacional

Sistema industrial *End-to-End* para diagnóstico preditivo de modos de falha em turbinas e garantia multimodal de qualidade em montagem mecânica por Visão Computacional e telemetria IoT em nós de Edge Computing.

<br>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/STATUS-CONCLU%C3%8DDO-2ea44f?style=for-the-badge)
![License](https://img.shields.io/badge/LICENSE-MIT-yellow?style=for-the-badge)

</div>

---

## 1. Visão Geral e Objetivos do Projeto

A **AeroMotion Dynamics S.A.** instrumentou quatro bancadas de testes (equipamentos `TURB-101` a `TURB-104`) com sensores IoT amostrados a cada 10 minutos. Este projeto implanta uma infraestrutura analítica de ponta a ponta voltada a:

1. **Engenharia de Dados em Séries Temporais:** Saneamento contínuo de ruídos de interferência eletromagnética (EMI), imputação por interpolação física linear e extração de *rolling features* e severidade dinâmica.
2. **Inteligência Preditiva Multiclasse:** Classificação antecipada de falhas (Desgaste de Rolamento, Desbalanceamento de Rotor e Cavitação/Superaquecimento) e detecção não supervisionada de desvios operacionais com Isolation Forest.
3. **Controle Óptico e Térmico (Visão Computacional):** Avaliação de circularidade e desgaste superficial de componentes via Canny Edge Detection, e isolamento de *hotspots* infravermelhos no espaço de cor HSV.
4. **Deploy Industrial:** Microsserviço RESTful assíncrono em FastAPI, dashboard web operacional com tolerâncias dinâmicas e conteinerização em Docker para nós de borda.

---

## 2. Estrutura do Repositório

```text
Projeto-Integrador-Aero/
├── app/
│   ├── __init__.py
│   └── main.py                     # API RESTful FastAPI e Dashboard Industrial
├── data/
│   ├── dados_sensores_tratados.csv # Dataset pós-saneamento e feature engineering
│   └── raw/                        # Diretório para dados brutos (telemetria IoT)
├── docs/
│   ├── images/                     # Evidências geradas pelo pipeline de visão
│   ├── Parecer_tecnico.md          # Parecer regulatório (LGPD, EU AI Act, Cibersegurança)
│   └── relatorio_tecnico_mlops.md  # Relatório analítico, EDA e plano de MLOps
├── models/
│   ├── best_model.joblib           # Classificador multiclasse treinado (XGBoost/RF)
│   ├── isolation_forest.joblib     # Detector não supervisionado de anomalias
│   └── preprocessor.joblib         # Pipeline de pré-processamento (RobustScaler + OneHot)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Rotinas de saneamento e rolling features
│   ├── train_model.py              # Pipeline de treino, validação e split temporal
│   └── vision_inspection.py        # Algoritmos de Canny, circularidade e HSV
├── Tratamento.ipynb                # Notebook de exploração estrutural e EDA
├── Dockerfile                      # Build enxuto da imagem para Edge Computing
├── docker-compose.yml              # Orquestração do serviço de inferência
├── requirements.txt                # Dependências versionadas do projeto
└── README.md

```

---

## 3. Tecnologias e Bibliotecas

* **Linguagem & Ambiente:** Python 3.10+, Visual Studio Code
* **Engenharia e Análise de Dados:** Pandas, NumPy, SciPy
* **Machine Learning & MLOps:** Scikit-Learn, XGBoost, Imbalanced-Learn, Joblib
* **Visão Computacional:** OpenCV (`opencv-python-headless`)
* **Deploy & Infraestrutura:** FastAPI, Pydantic V2, Uvicorn, Docker, Docker Compose

---

## 4. Guia de Instalação e Execução

### Pré-requisitos

* Python 3.10 ou superior instalado
* Git instalado

### Passo 1: Clonar o Repositório

```bash
git clone [https://github.com/henrique-svg21/Projeto-Integrador-Aero.git](https://github.com/henrique-svg21/Projeto-Integrador-Aero.git)
cd Projeto-Integrador-Aero

```

### Passo 2: Configurar o Ambiente Virtual (`venv`)

* No Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```


* No Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate

```



### Passo 3: Instalar as Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

---

## 5. Pipeline de Execução dos Módulos

### 1. Saneamento e Engenharia de Features

Abra e execute sequencialmente as células do arquivo `Tratamento.ipynb` no VS Code ou processe via script para gerar `data/dados_sensores_tratados.csv`.

### 2. Treinamento dos Modelos de Machine Learning

Execute o script de treinamento para calibrar os modelos supervisionados e não supervisionados e exportar os pesos para a pasta `models/`:

```bash
python src/train_model.py

```

### 3. Validação dos Algoritmos de Visão Computacional

Para testar os limites do Canny Edge Detection, circularidade e segmentação térmica em terminal:

```bash
python src/vision_inspection.py

```

---

## 6. Inicialização do Microsserviço & Dashboard

Inicie a aplicação com o servidor Uvicorn a partir da raiz do projeto:

```bash
uvicorn app.main:app --reload --port 8000

```

### Interfaces Disponíveis:

* **Central de Controle & Metrologia (Dashboard Web):**
Acesse `http://127.0.0.1:8000` no seu navegador.
Permite carregar imagens de referência e teste, ajustar sliders de tolerância industrial (desgaste aceitável, circularidade mínima, filtros de luz) e testar leituras dos sensores em tempo real.
* **Documentação Interativa da API (Swagger UI):**
Acesse `http://127.0.0.1:8000/docs` para inspecionar schemas e realizar requisições REST diretamente.

---

## 7. Validação da API via Linha de Comando (PowerShell)

Abra uma segunda aba de terminal (com o servidor ativo) e teste as rotas principais:

### 1. Verificação de Saúde (`GET /health`)

```powershell
Invoke-RestMethod -Uri "[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)" -Method GET

```

### 2. Diagnóstico Preditivo de Operação Nominal (`POST /predict`)

```powershell
$body = @{
    id_equipamento = "TURB-101"
    modo_operacao = "Cruzeiro"
    horas_operacao_acumuladas = 500.0
    temperatura_mancal_c = 58.5
    vibracao_rms_mms = 2.1
    pressao_oleo_bar = 4.2
    rotacao_rpm = 2950.0
    emissao_acustica_db = 48.0
    corrente_eletrica_amp = 26.5
} | ConvertTo-Json

Invoke-RestMethod -Uri "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" -Method POST -Body $body -ContentType "application/json"

```

### 3. Diagnóstico de Falha Crítica (`POST /predict`)

```powershell
$bodyFalha = @{
    id_equipamento = "TURB-103"
    modo_operacao = "Carga_Alta"
    horas_operacao_acumuladas = 2100.0
    temperatura_mancal_c = 72.0
    vibracao_rms_mms = 9.8
    pressao_oleo_bar = 4.3
    rotacao_rpm = 3400.0
    emissao_acustica_db = 55.0
    corrente_eletrica_amp = 38.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" -Method POST -Body $bodyFalha -ContentType "application/json"

```

---

## 8. Desempenho dos Modelos Preditivos

Resultados obtidos no conjunto de teste independente (Split Temporal de 30% mais recente):

| Modelo | Classe de Falha | Precisão | Recall | F1-Score | Acurácia Global |
| --- | --- | --- | --- | --- | --- |
| **Random Forest** | 0 (Normal) | 99,62% | 99,91% | 99,76% | **99,47%** |
|  | 1 (Desgaste Rolamento) | 97,73% | 93,99% | 95,82% |  |
|  | 2 (Desbalanceamento Rotor) | 98,01% | 98,01% | 98,01% |  |
|  | 3 (Cavitação/Superaquec.) | 100,00% | 98,13% | 99,06% |  |
| **XGBoost Classifier** | 0 (Normal) | 99,72% | 99,97% | 99,84% | **99,67%** |
|  | 1 (Desgaste Rolamento) | 100,00% | 94,54% | 97,19% |  |
|  | 2 (Desbalanceamento Rotor) | 98,68% | 99,34% | 99,01% |  |
|  | 3 (Cavitação/Superaquec.) | 99,07% | 100,00% | 99,53% |  |

* **Destaque:** Ambas as abordagens superaram a meta de projeto de **Recall > 90%** em todas as categorias de falha. O **XGBoost Classifier** foi implementado como modelo produtivo final por atingir **100% de Recall na classe de Cavitação/Superaquecimento**.

---

## 9. Governança, Regulamentação & MLOps

* **LGPD (Lei 13.709/2018):** As câmeras de chão de fábrica operam com protocolo de *Privacy by Design* e rotinas de anonimização (Face Blurring) processadas localmente antes do envio de métricas.
* **EU AI Act:** O sistema foi categorizado como **IA de Alto Risco (High-Risk)**, incorporando registro contínuo de logs auditáveis e supervisão humana mandatória (*Human-in-the-Loop*) para bloqueios de máquinas.
* **Cibersegurança OT/IT (ISA/IEC 62443):** O nó de inferência opera em DMZ Industrial com autenticação por mTLS e proteção estrita contra envenenamento de dados (*Data Poisoning*).
* **Plano de MLOps:** Monitoramento de *Data Drift* baseado em testes Kolmogorov-Smirnov e rotina de retreinamento trimestral integrada ao detector de anomalias não supervisionado Isolation Forest.

---

## 10. Implantação em Edge Computing (Docker)

Caso o nó fabril possua suporte a contêineres:

```bash
docker-compose up --build -d

```

O serviço será exposto na porta `8000` com reinicialização automática e verificação contínua de integridade via `/health`.
