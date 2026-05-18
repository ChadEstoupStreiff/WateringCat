import io
from time import sleep
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from picamera2 import Picamera2

app = FastAPI()
camera = Picamera2()
camera.configure(camera.create_still_configuration())
camera.start()


def turn_on_pump():
    pass # TODO: Implement pump control logic here

def turn_off_pump():
    pass # TODO: Implement pump control logic here

@app.get("/shot")
def shot():
    try:
        buf = io.BytesIO()
        camera.capture_file(buf, format="jpeg")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
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