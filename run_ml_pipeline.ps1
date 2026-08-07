python -m src.download_data

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

docker compose up -d --build

docker compose run --rm api python -m src.prepare_data
docker compose run --rm api python -m src.split_data
docker compose run --rm api python -m src.train_baseline
docker compose run --rm api python -m src.tune_model --trials 8 --metric f1
docker compose run --rm api python -m src.train_production
docker compose run --rm api python -m src.evaluate_model
docker compose run --rm api python -m src.register_model
