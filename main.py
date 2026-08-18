```python
import os
import io
import numpy as np

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, ImageOps

import tensorflow as tf
from mtcnn import MTCNN

from supabase import create_client


# ============================================================
# APPLICATION FASTAPI
# ============================================================

app = FastAPI()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# ============================================================
# MODÈLE ET DÉTECTEUR DE VISAGE
# ============================================================

MODEL_PATH = "facenet_quant.tflite"

interpreter = None

detector = MTCNN()


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

def load_model():
    global interpreter

    if interpreter is None:
        interpreter = tf.lite.Interpreter(
            model_path=MODEL_PATH
        )
        interpreter.allocate_tensors()

    return interpreter


# ============================================================
# CRÉATION DE L'EMBEDDING
# ============================================================

def get_embedding(image_bytes: bytes):

    # Ouvrir l'image
    img = Image.open(
        io.BytesIO(image_bytes)
    )

    # Correction de l'orientation EXIF.
    # Important pour les photos prises avec les téléphones Android.
    img = ImageOps.exif_transpose(img)

    # Conversion en RGB
    img = img.convert("RGB")

    # Conversion en tableau NumPy
    img_array = np.array(img)

    # Détection des visages
    faces = detector.detect_faces(img_array)

    if not faces:
        return None

    # Premier visage détecté
    x, y, w, h = faces[0]["box"]

    # Éviter les coordonnées négatives
    x = max(0, x)
    y = max(0, y)

    # Recadrage du visage
    face = img_array[
        y:y + h,
        x:x + w
    ]

    # Redimensionnement pour FaceNet
    face_img = Image.fromarray(face).resize(
        (160, 160)
    )

    # Conversion en float32
    face_array = np.array(
        face_img
    ).astype(np.float32)

    # Normalisation FaceNet
    face_array = (
        face_array - 127.5
    ) / 128.0

    # Ajouter la dimension batch
    face_array = np.expand_dims(
        face_array,
        axis=0
    )

    # Charger le modèle
    interp = load_model()

    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Envoyer l'image au modèle
    interp.set_tensor(
        input_details[0]["index"],
        face_array
    )

    # Exécuter le modèle
    interp.invoke()

    # Récupérer l'embedding
    embedding = interp.get_tensor(
        output_details[0]["index"]
    )[0]

    return embedding


# ============================================================
# ENRÔLEMENT D'UN ÉLÈVE
# ============================================================

@app.post("/enroll")
async def enroll(
    student_id: str,
    file: UploadFile = File(...)
):

    # Lire l'image
    image_bytes = await file.read()

    # Créer l'embedding
    embedding = get_embedding(
        image_bytes
    )

    # Aucun visage détecté
    if embedding is None:

        return {
            "success": False,
            "message": "Aucun visage détecté"
        }

    # Enregistrer l'embedding dans Supabase
    supabase.table(
        "students"
    ).update(
        {
            "face_embedding": embedding.tolist()
        }
    ).eq(
        "id",
        student_id
    ).execute()

    return {
        "success": True
    }


# ============================================================
# RECONNAISSANCE FACIALE
# ============================================================

@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...)
):

    # Lire l'image
    image_bytes = await file.read()

    # Créer l'embedding du visage
    embedding = get_embedding(
        image_bytes
    )

    # Aucun visage détecté
    if embedding is None:

        print(
            "Aucun visage détecté"
        )

        return {
            "student_id": None,
            "message": "Aucun visage détecté"
        }

    # Récupérer les élèves ayant un embedding
    response = supabase.table(
        "students"
    ).select(
        "id, face_embedding"
    ).not_.is_(
        "face_embedding",
        "null"
    ).execute()

    students = response.data

    # Variables de recherche
    best_id = None
    best_distance = float("inf")

    # Seuil actuel
    threshold = 0.9

    # Comparaison avec chaque élève
    for student in students:

        stored = np.array(
            student["face_embedding"],
            dtype=np.float32
        )

        distance = np.linalg.norm(
            embedding - stored
        )

        if distance < best_distance:

            best_distance = distance
            best_id = student["id"]

    # Logs pour le diagnostic
    print(
        "BEST STUDENT:",
        best_id
    )

    print(
        "BEST DISTANCE:",
        best_distance
    )

    print(
        "THRESHOLD:",
        threshold
    )

    # Vérifier si la distance est suffisamment faible
    if (
        best_id is not None
        and best_distance < threshold
    ):

        return {
            "student_id": best_id
        }

    # Aucun élève reconnu
    return {
        "student_id": None
    }


# ============================================================
# TEST DE L'API
# ============================================================

@app.get("/")
async def health():

    return {
        "status": "ok"
    }
```
