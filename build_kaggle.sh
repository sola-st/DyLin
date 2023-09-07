# Usage: bash build_kaggle.sh <competition_id>
docker build -f Dockerfile.kaggle --build-arg kaggle_competition=$1 -t dylin_kaggle .