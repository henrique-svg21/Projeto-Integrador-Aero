# Imagem base oficial enxuta com Python 3.10
FROM python:3.10-slim

# Variáveis de ambiente para performance do Python em containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalação de dependências do sistema necessárias para OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalação de requisitos Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cópia do código-fonte e artefatos
COPY src/ /app/src/
COPY app/ /app/app/
COPY models/ /app/models/

# Exposição da porta de comunicação
EXPOSE 8000

# Execução assíncrona com Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]