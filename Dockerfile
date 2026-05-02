# ============================================================
# DOCKERFILE — Credit Card Fraud Detection ML System
# ============================================================

FROM python:3.10.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# libgomp1 required by LightGBM / XGBoost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_api.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_api.txt


FROM python:3.10.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY src/              ./src/
COPY serving/          ./serving/
COPY services/         ./services/
COPY feature_store/    ./feature_store/
COPY graph_detection/  ./graph_detection/
COPY fraud_models/     ./fraud_models/
COPY logs/             ./logs/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "serving.fraud_api:app", "--host", "0.0.0.0", "--port", "8000"]
