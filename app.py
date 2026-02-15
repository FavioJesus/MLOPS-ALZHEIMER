import io, os
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse


MODEL_PATH = os.getenv("MODEL_PATH", "/app/model.h5")


CLASS_NAMES = [
    'Leve Demencia',        
    'Moderada Demencia',   
    'Sin Demencia',       
    'Demencia Alta'     
]


app = FastAPI(title="Image Inference API", version="1.0")


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax'),
    ])

# construye y carga PESOS
model = build_model()
try:
    model.load_weights(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"No pude cargar pesos desde {MODEL_PATH}: {e}")

def preprocess(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))  # el modelo espera 224x224x3
    arr = np.asarray(img).astype("float32") 
    return arr[None, ...]  # (1,224,224,3)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Solo JPG/PNG/WebP")
    data = await file.read()
    x = preprocess(data)
    preds = model.predict(x)  # ya sale PROBABILIDADES porque la última capa es softmax
    probs = preds[0].tolist()
    top = int(np.argmax(probs))
    return JSONResponse({
        "label": CLASS_NAMES[top],
        "top_index": top,
        "probs": probs,
         "class_names": CLASS_NAMES  
    })

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="static")
