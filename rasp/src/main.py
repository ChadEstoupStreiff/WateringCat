import io
from time import sleep
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
<<<<<<< HEAD
import RPi.GPIO as GPIO
import cv2
=======
from picamera2 import Picamera2
import RPi.GPIO as GPIO
>>>>>>> 88dd0cda70126c2d14b37bc1ce576108fbd9d244

RELAY_PIN = 17

GPIO.setmode(GPIO.BCM)
<<<<<<< HEAD
GPIO.setwarnings(False)
=======
>>>>>>> 88dd0cda70126c2d14b37bc1ce576108fbd9d244
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.HIGH)  # Pompe OFF par défaut

app = FastAPI()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Could not open camera on /dev/video0")


def capture_photo() -> bytes:
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture frame")
    _, buf = cv2.imencode('.jpg', frame)
    return buf.tobytes()


def turn_on_pump():
    GPIO.output(RELAY_PIN, GPIO.LOW)  # Actif LOW → pompe ON


def turn_off_pump():
    GPIO.output(RELAY_PIN, GPIO.HIGH)  # Pompe OFF


@app.on_event("shutdown")
def shutdown():
    cap.release()
    GPIO.cleanup()


def turn_on_pump():
    GPIO.output(RELAY_PIN, GPIO.LOW)  # Actif LOW → pompe ON

def turn_off_pump():
    GPIO.output(RELAY_PIN, GPIO.HIGH)  # Pompe OFF

@app.get("/shot")
def shot():
    try:
        jpeg_bytes = capture_photo()
        return StreamingResponse(io.BytesIO(jpeg_bytes), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pump/on")
def pump_on(duration: int = None):
    try:
        turn_on_pump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if duration is not None:
        sleep(duration)
        try:
            turn_off_pump()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/pump/off")
def pump_off():
    try:
        turn_off_pump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pump/on")
def pump_on(duration: int = None):
    try:
        turn_on_pump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if duration is not None:
        sleep(duration)
        try:
            turn_off_pump()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/pump/off")
def pump_off():
    try:
        turn_off_pump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))