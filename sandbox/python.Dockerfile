# Sandbox Python : pytest uniquement, aucun réseau à l'exécution (--network none).
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.3.3 \
 && useradd -m runner
ENV PYTHONDONTWRITEBYTECODE=1
USER runner
