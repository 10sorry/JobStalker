FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev libssl-dev build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
WORKDIR /app/src

# точка монтирования для session и sqlite
VOLUME ["/data"]

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
