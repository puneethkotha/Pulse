#!/usr/bin/env bash
# Download and extract the MovieLens 1M dataset.
#
# Source: https://files.grouplens.org/datasets/movielens/ml-1m.zip
# License: https://grouplens.org/datasets/movielens/ (non-commercial use)

set -euo pipefail

DATA_DIR="data/raw"
TARGET_DIR="${DATA_DIR}/ml-1m"
ZIP_PATH="${DATA_DIR}/ml-1m.zip"
URL="https://files.grouplens.org/datasets/movielens/ml-1m.zip"

mkdir -p "${DATA_DIR}"

if [ -d "${TARGET_DIR}" ] && [ -f "${TARGET_DIR}/ratings.dat" ]; then
    echo "Dataset already present at ${TARGET_DIR}. Skipping download."
    exit 0
fi

echo "Downloading MovieLens 1M from ${URL} ..."
curl -L -o "${ZIP_PATH}" "${URL}"

echo "Extracting ..."
unzip -q "${ZIP_PATH}" -d "${DATA_DIR}"
rm "${ZIP_PATH}"

echo "Done. Dataset at ${TARGET_DIR}"
echo "Files:"
ls "${TARGET_DIR}"
