FROM python:3.11-slim

# Системные зависимости:
# fonts-dejavu-core — шрифты с кириллицей для Linux (замена Segoe UI)
# libgomp1          — требуется для многопоточности Numba
# tini              -- менеджер процессов (PID 1), корректно завершает дочерние процессы
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        libgomp1 \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем оба файла зависимостей
COPY requirements.txt requirements-visualization.txt ./

# Устанавливаем сначала базовые зависимости, затем для визуализации
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-visualization.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Используем tini как entrypoint для корректной обработки сигналов завершения
ENTRYPOINT ["tini", "--", "python"]
CMD ["run.py", "--help"]]