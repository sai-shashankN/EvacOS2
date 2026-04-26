FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY requirements.txt /app/requirements.txt
COPY openenv.yaml /app/openenv.yaml
COPY evacos_ma /app/evacos_ma
COPY procgen /app/procgen
COPY server /app/server

RUN pip install --upgrade pip && pip install .

CMD ["sh", "-c", "uvicorn evacos_ma.api:app --host 0.0.0.0 --port ${PORT}"]
