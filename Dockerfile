FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PORT=7860

WORKDIR /app

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
        "fastapi>=0.115,<1" \
        "uvicorn>=0.30,<1" \
        "pydantic>=2.7,<3" \
        "numpy>=1.26,<3" \
        "pyyaml>=6,<7" \
        "openenv-core>=0.2.0"

COPY openenv.yaml /app/openenv.yaml
COPY evacos_ma /app/evacos_ma
COPY procgen /app/procgen

EXPOSE 7860

CMD ["sh", "-c", "uvicorn evacos_ma.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
