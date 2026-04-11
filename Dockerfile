FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libzbar0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "gunicorn main:app --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 120"]
