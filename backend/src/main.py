import base64
import io
import logging

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from core import CANVAS_H, CANVAS_W, Backend, compute_first_mask_iou, save_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="WateringCat Backend")
backend = Backend()


@app.get("/status")
def get_status():
    return {
        "cat_tolerance": backend.cat_tolerance,
        "iou_tolerance": backend.iou_tolerance,
        "pump_duration": backend.pump_duration,
        "pulling_interval": backend.pulling_interval,
        "model_name": backend.model_name,
    }


@app.get("/photo")
def get_photo():
    try:
        photo = backend.pull_photo(force_shape=(CANVAS_W, CANVAS_H))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    buf = io.BytesIO()
    Image.fromarray(photo).save(buf, format="JPEG")
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.post("/pump")
def activate_pump(duration: int):
    try:
        backend.activate_pump(duration)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"ok": True}


@app.get("/mask")
def get_mask():
    buf = io.BytesIO()
    Image.fromarray(backend.activation_mask.astype(np.uint8) * 255).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.put("/mask")
async def update_mask(file: UploadFile = File(...)):
    data = await file.read()
    mask_arr = np.array(Image.open(io.BytesIO(data)).convert("L")) > 0
    backend.activation_mask = mask_arr
    save_mask(mask_arr, "/app/config/activation_mask.png")
    return {"ok": True}


@app.get("/events")
def get_events(limit: int = 500):
    return backend.event_log.get_events(limit)


@app.delete("/events")
def clear_events():
    backend.event_log.clear()
    return {"ok": True}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    photo = np.array(Image.open(io.BytesIO(data)).convert("RGB"))

    detection = backend.detect_cat(photo)
    if detection is None:
        return {"detected": False, "class_name": None, "confidence": 0.0, "iou": 0.0, "cat_mask_png": None}

    cat_mask, confidence, class_name = detection
    iou = compute_first_mask_iou(cat_mask, backend.activation_mask)

    buf = io.BytesIO()
    Image.fromarray(cat_mask.astype(np.uint8) * 255).save(buf, format="PNG")
    cat_mask_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "detected": True,
        "class_name": class_name,
        "confidence": confidence,
        "iou": iou,
        "cat_mask_png": cat_mask_b64,
    }
