@@ -1,113 +1,114 @@
import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://educonnectniger.flutterflow.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from PIL import Image
import tensorflow as tf
from mtcnn import MTCNN
from supabase import create_client

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_PATH = "facenet_model"
interpreter = None
detector = MTCNN()

def load_model():
    global interpreter
    if interpreter is None:
        interpreter = tf.lite.Interpreter(model_path="facenet_quant.tflite")
        interpreter.allocate_tensors()
    return interpreter

def get_embedding(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    faces = detector.detect_faces(img_array)
    if not faces:
        return None

    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)
    face = img_array[y:y+h, x:x+w]

    face_img = Image.fromarray(face).resize((160, 160))
    face_array = np.array(face_img).astype(np.float32)
    face_array = (face_array - 127.5) / 128.0
    face_array = np.expand_dims(face_array, axis=0)

    interp = load_model()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    interp.set_tensor(input_details[0]['index'], face_array)
    interp.invoke()
    embedding = interp.get_tensor(output_details[0]['index'])[0]

    return embedding

@app.post("/enroll")
async def enroll(student_id: str, file: UploadFile = File(...)):
    image_bytes = await file.read()
    embedding = get_embedding(image_bytes)

    if embedding is None:
        return {"success": False, "message": "Aucun visage détecté"}

    supabase.table("students").update(
        {"face_embedding": embedding.tolist()}
    ).eq("id", student_id).execute()

    return {"success": True}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    image_bytes = await file.read()
    embedding = get_embedding(image_bytes)

    if embedding is None:
        return {"student_id": None, "message": "Aucun visage détecté"}

    response = supabase.table("students").select("id, face_embedding").not_.is_("face_embedding", "null").execute()
    students = response.data

    best_id = None
    best_distance = float("inf")
    threshold = 0.9

    for student in students:
        stored = np.array(student["face_embedding"])
        distance = np.linalg.norm(embedding - stored)
        if distance < best_distance:
            best_distance = distance
            best_id = student["id"]

    if best_id is not None and best_distance < threshold:
        return {"student_id": best_id}

    return {"student_id": None}

@app.get("/")
async def health():
    return {"status": "ok"}
