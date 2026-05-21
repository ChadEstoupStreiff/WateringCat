import asyncio
import io
import logging
import threading
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import RPi.GPIO as GPIO
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wateringcat")

RELAY_PIN = 17

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)  # Pompe OFF par défaut

app = FastAPI()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize stale-frame buffer

if not cap.isOpened():
    raise RuntimeError("Could not open camera on /dev/video0")
log.info("Camera opened at 1280x720 MJPG")

# Latest frame kept by background grabber thread
_latest_frame: bytes | None = None
_frame_lock = threading.Lock()


def _frame_grabber():
    global _latest_frame
    log.info("Frame grabber thread started")
    while True:
        ret, frame = cap.read()
        if ret:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            h = frame.shape[0]
            cv2.putText(frame, ts, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with _frame_lock:
                _latest_frame = buf.tobytes()
        elif not ret:
            log.warning("Frame grab failed")


_grabber_thread = threading.Thread(target=_frame_grabber, daemon=True)
_grabber_thread.start()


def turn_on_pump():
    GPIO.output(RELAY_PIN, GPIO.HIGH)  # Actif LOW → pompe ON
    log.info("Pump ON")


def turn_off_pump():
    GPIO.output(RELAY_PIN, GPIO.LOW)  # Pompe OFF
    log.info("Pump OFF")


@app.on_event("shutdown")
def shutdown():
    log.info("Shutting down — releasing camera and GPIO")
    cap.release()
    GPIO.cleanup()


@app.get("/shot")
def shot():
    with _frame_lock:
        frame = _latest_frame
    if frame is None:
        log.warning("Shot requested but camera not ready")
        raise HTTPException(status_code=503, detail="Camera not ready yet")
    log.info("Shot served (%d bytes)", len(frame))
    return StreamingResponse(io.BytesIO(frame), media_type="image/jpeg")


@app.get("/pump/on")
async def endpoint_pump_on(duration: int | None = None):
    try:
        turn_on_pump()
    except Exception as e:
        log.error("Failed to turn pump on: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    if duration is not None:
        log.info("Pump will auto-off in %ds", duration)
        await asyncio.sleep(duration)
        try:
            turn_off_pump()
        except Exception as e:
            log.error("Failed to auto-off pump: %s", e)
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/pump/off")
def endpoint_pump_off():
    try:
        turn_off_pump()
    except Exception as e:
        log.error("Failed to turn pump off: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/temperature")
def get_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp_celsius = int(f.read().strip()) / 1000.0
        return {"temperature": temp_celsius}
    except Exception as e:
        log.error("Failed to read CPU temperature: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
