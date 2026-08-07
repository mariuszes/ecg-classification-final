PYTHON = python
DC = docker compose
RUN = $(DC) run --rm api

.PHONY: up down build download-data prepare split train-baseline tune train-production evaluate register full-pipeline logs

up:
	$(DC) up -d --build

down:
	$(DC) down

build:
	$(DC) build

download-data:
	$(PYTHON) -m src.download_data

prepare:
	$(RUN) $(PYTHON) -m src.prepare_data

split:
	$(RUN) $(PYTHON) -m src.split_data

train-baseline:
	$(RUN) $(PYTHON) -m src.train_baseline

tune:
	$(RUN) $(PYTHON) -m src.tune_model --trials 8 --metric f1

train-production:
	$(RUN) $(PYTHON) -m src.train_production

evaluate:
	$(RUN) $(PYTHON) -m src.evaluate_model

register:
	$(RUN) $(PYTHON) -m src.register_model

full-pipeline: download-data prepare split train-baseline tune train-production evaluate register

logs:
	$(DC) logs -f api frontend mlflow
