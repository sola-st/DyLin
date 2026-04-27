docker build -f Dockerfile.kaggle --build-arg kaggle_competition=$1 -t dylin_kaggle .  # $1 = competition id (e.g. titanic); needs kaggle.json in repo root
