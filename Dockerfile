FROM python:3.11-slim

WORKDIR /app

# Install system libs
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🔥 PRE-DOWNLOAD EASYOCR MODEL
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Copy app
COPY . .

# Start app
CMD ["sh", "-c", "gunicorn main:app --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 120"]
