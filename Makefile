.PHONY: fetch clean train sweep compare test api up pipeline
fetch:       ## скачать наборы с data.mos.ru в data/raw (нужен MOS_API_KEY в .env)
	python -m src.ingest
clean:       ## очистка + DQ-отчёт + ворота качества; Spark: make clean ENGINE=spark
	python -m src.training --stage data --engine $(or $(ENGINE),pandas)
train:       ## бейзлайны + GBM + PyTorch -> детекторы аномалий -> MLflow -> artifacts/latest
	python -m src.training --stage train
sweep:       ## перебор гиперпараметров
	python -m src.training --stage sweep
compare:
	python -m src.training --stage compare
test:
	python -m pytest -q
api:
	uvicorn api.main:app --reload
up:          ## PostgreSQL + MLflow в Docker
	docker compose up -d postgres mlflow
pipeline:    ## полный пайплайн в Docker: API -> PostgreSQL -> Spark -> обучение
	docker compose --profile jobs run --rm pipeline
