FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
# copia tu modelo al contenedor (conservando estructura)
COPY Recursos/alz_model_trained.h5 /app/model.h5
COPY static ./static     
# define la ruta del modelo dentro del contenedor
ENV MODEL_PATH=/app/model.h5
# (opcional) nombres de clases
# ENV CLASS_NAMES=gato,perro,ave

EXPOSE 8080
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
