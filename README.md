# 🧠 Red Neuronal para Predicción de Alzheimer

**Maestría en Data Science -- UNI**\
**Alumno:** Guevara Puente, Favio Jesús

------------------------------------------------------------------------

## 🌍 Aplicación Desplegada

La aplicación se encuentra desplegada en la nube mediante **Render**:

🔗 **URL pública:**\
https://prediccion-alzheimer.onrender.com/

⚠️ **Nota sobre disponibilidad:**\
El sistema está desplegado en un **plan gratuito de Render**.\
Esto implica que:

-   Si la aplicación no recibe peticiones durante un período de tiempo,
    el servicio se suspende automáticamente.
-   Cuando un usuario vuelve a ingresar, el sistema puede tardar **1 a 3
    minutos en levantarse nuevamente**.
-   Luego de iniciarse, funciona con normalidad.

El sistema está operativo y funcionando correctamente.

------------------------------------------------------------------------

## 🔄 Despliegue Automático (CI/CD)

El proyecto está conectado directamente a GitHub.

Render permite:

-   Integración automática con repositorio GitHub.
-   Deploy automático cada vez que se realiza un commit a la rama
    configurada.
-   Actualización automática del contenedor sin intervención manual.

Flujo:

    Commit → Push a GitHub → Render detecta cambios → Build automático → Deploy automático

Esto garantiza:

-   Versionamiento controlado
-   Actualizaciones rápidas
-   Entorno reproducible mediante Docker

------------------------------------------------------------------------

## 📌 Descripción del Proyecto

Este proyecto implementa un sistema basado en **Deep Learning** capaz de
realizar un **pre-diagnóstico de Alzheimer** a partir de imágenes
médicas cerebrales.

El sistema:

-   Entrena un modelo en Google Colab
-   Exporta los pesos entrenados a un archivo `.h5`
-   Implementa una API con FastAPI
-   Despliega una interfaz web para cargar imágenes
-   Devuelve la predicción y probabilidades por clase

⚠️ **Importante:**\
Este sistema es de apoyo diagnóstico y **no reemplaza una evaluación
médica profesional**.

------------------------------------------------------------------------

# 🎯 Objetivos

## Objetivo General

Desarrollar un modelo de Deep Learning capaz de identificar y clasificar
la presencia y etapa del Alzheimer a partir de imágenes médicas,
utilizando técnicas de aprendizaje profundo para mejorar la eficiencia y
precisión del sistema.

## Objetivos Específicos

1.  Implementar procesos de tratamiento y depuración de imágenes.
2.  Estandarizar las variables y preparar los datos para el modelado.
3.  Aplicar algoritmos de Deep Learning.
4.  Evaluar el modelo utilizando métricas de clasificación:
    -   Accuracy
    -   Matriz de confusión
    -   F1-score
    -   ROC-AUC
5.  Implementar una API para inferencia en tiempo real.
6.  Desplegar una interfaz web para interacción del usuario.

------------------------------------------------------------------------

# 🧪 Dataset y Clases

## Dataset utilizado
El dataset utilizado se encuentra en este link: https://drive.google.com/drive/folders/1-BQ-IL3gz71_FRJkNdZ2ifibkOBQyV4q?usp=sharing
Dentro hay 2 carpetas train y test. 



## Modelo

El modelo clasifica imágenes en cuatro categorías:

  Clase               Descripción
  ------------------- ----------------------------
  Leve Demencia       Demencia en etapa temprana
  Moderada Demencia   Demencia intermedia
  Sin Demencia        Paciente sano
  Demencia Alta       Demencia severa

Las imágenes fueron redimensionadas a:

    224 x 224 x 3 (RGB)

------------------------------------------------------------------------

# 🧠 Entrenamiento del Modelo

El entrenamiento fue realizado en Google Colab:

Notebook original:\
https://colab.research.google.com/drive/1DFOYaH3a9eKJz0i1l4a5BRZH7XPIotNz

## Arquitectura del Modelo

Para producción se utiliza la siguiente arquitectura:

    tf.keras.Sequential([
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
        tf.keras.layers.Dense(4, activation='softmax')
    ])

------------------------------------------------------------------------

## 📊 Resultados Obtenidos

Durante el entrenamiento:

-   Accuracy final entrenamiento ≈ 97.7%
-   Validation Accuracy final ≈ 95.62%
-   Validation Loss final ≈ 0.1335

Estos resultados indican una buena capacidad de generalización del
modelo.

------------------------------------------------------------------------

# 🖥 Arquitectura del Sistema

    Usuario → Frontend (HTML + JS)
                 ↓
              FastAPI
                 ↓
           TensorFlow Model (.h5)
                 ↓
            Deploy en Render (Docker)

------------------------------------------------------------------------

# 🌐 Frontend

Ubicación: `static/index.html`

Características:

-   Drag & Drop de imágenes
-   Validación de tipo de archivo
-   Consumo del endpoint `/predict`
-   Visualización de probabilidades por clase
-   Modal informativo inicial
-   Diseño responsive moderno

------------------------------------------------------------------------

# ⚙️ Backend (FastAPI)

Archivo principal: `app.py`

## Endpoints

### GET /health

Verifica que la API esté activa.

Respuesta:

    { "status": "ok" }

### POST /predict

Recibe una imagen y devuelve la predicción.

Formato de respuesta:

    {
      "label": "Leve Demencia",
      "top_index": 0,
      "probs": [0.12, 0.80, 0.05, 0.03],
      "class_names": [
        "Leve Demencia",
        "Moderada Demencia",
        "Sin Demencia",
        "Demencia Alta"
      ]
    }

------------------------------------------------------------------------

# 🐳 Dockerización

Dockerfile:

-   Imagen base: python:3.11-slim
-   Instalación de dependencias
-   Copia del modelo entrenado (.h5)
-   Exposición del puerto 8080

Ejecución:

    docker build -t alzheimer-app .
    docker run -p 8080:8080 alzheimer-app

------------------------------------------------------------------------

# 📦 Dependencias

    fastapi==0.115.0
    uvicorn[standard]==0.30.6
    pillow==10.4.0
    numpy==1.26.4
    tensorflow-cpu==2.15.0
    python-multipart==0.0.9
    python-dotenv==1.0.1

------------------------------------------------------------------------

# 🚀 Ejecución Local

1.  Instalar dependencias:

```{=html}
<!-- -->
```
    pip install -r requirements.txt

2.  Ejecutar servidor:

```{=html}
<!-- -->
```
    uvicorn app:app --reload

3.  Abrir en navegador:

```{=html}
<!-- -->
```
    http://localhost:8000

------------------------------------------------------------------------

# 📌 Consideraciones Finales

-   El modelo fue entrenado en Google Colab.
-   Se exportaron los pesos entrenados en formato `.h5`.
-   El sistema permite inferencia en tiempo real.
-   La aplicación está desplegada en Render con CI/CD automático.
-   Es un prototipo académico.
-   No sustituye diagnóstico médico profesional.

------------------------------------------------------------------------

# 👨‍🎓 Autor

Guevara Puente, Favio Jesús\
Maestría en Data Science -- UNI
