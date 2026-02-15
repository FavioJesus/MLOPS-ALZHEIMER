FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

COPY Recursos/alz_model_trained.h5 /app/model.h5
COPY static ./static     

ENV MODEL_PATH=/app/model.h5


EXPOSE 8080
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
