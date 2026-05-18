import io
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from picamera2 import Picamera2

app = FastAPI()
camera = Picamera2()
camera.configure(camera.create_still_configuration())
camera.start()


@app.get("/shot")
def shot():
    try:
        buf = io.BytesIO()
        camera.capture_file(buf, format="jpeg")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
