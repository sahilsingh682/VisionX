FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
RUN python get_models.py
EXPOSE 7860
CMD gunicorn -w 1 --threads 4 --timeout 180 -b 0.0.0.0:${PORT:-7860} app:app
