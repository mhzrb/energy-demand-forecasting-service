FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENERGY_RUNTIME=/app/runtime \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY pyproject.toml requirements.lock ./
COPY src ./src
COPY config ./config
RUN pip install -r requirements.lock && pip install --no-deps . && \
    useradd --uid 10001 --create-home app && \
    mkdir -p /app/runtime && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "energy.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
