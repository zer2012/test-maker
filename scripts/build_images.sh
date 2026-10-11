#!/bin/sh
# Construit les images de sandbox : ./scripts/build_images.sh [python javascript ...]
set -e
cd "$(dirname "$0")/.."
langs="${*:-python}"
for lang in $langs; do
  case "$lang" in
    python) image=testmaker-python ;;
    javascript) image=testmaker-node ;;
    *) image="testmaker-$lang" ;;
  esac
  docker build -t "$image" -f "sandbox/$lang.Dockerfile" sandbox
done
