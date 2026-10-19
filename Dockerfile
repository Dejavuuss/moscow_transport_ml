# Один образ для API и для пайплайна (Spark + обучение). Для продакшена его стоит разделить:
# API-образу не нужны Java и PySpark (образ станет в разы меньше).
# Debian зафиксирован явно: при смене базового дистрибутива тега "slim" (bookworm -> trixie)
# пропал пакет openjdk-17, и сборка падала.
FROM python:3.11-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MLFLOW_DISABLE_AGENT_HINT=1

# PySpark 4 требует Java 17+ (в trixie есть OpenJDK 21). curl — для healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-21-jre-headless curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Сначала зависимости, потом код: слой с зависимостями кэшируется, пока requirements.txt не менялся.
COPY requirements.txt .
# torch ставим отдельно из CPU-индекса: иначе pip может взять с PyPI сборку с CUDA (+несколько ГБ)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Не запускаем процесс от root. COPY --chown вместо chown -R: не дублирует слой с кодом.
RUN useradd -m app
COPY --chown=app:app . .
RUN mkdir -p data/raw data/processed artifacts && chown app:app data/raw data/processed artifacts
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
