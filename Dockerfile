# ── Backend – FastAPI / Uvicorn ──────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System libraries required by geopandas, psycopg2, scipy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
        libgdal-dev gdal-bin \
        libpq-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cache friendly)
COPY requirements.txt setup.py ./
COPY real_estate/ ./real_estate/
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e .

# Copy application code and data artefacts
COPY api/          ./api/
COPY artifact/     ./artifact/
COPY real_estate_data/ ./real_estate_data/
COPY inputs/       ./inputs/

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
