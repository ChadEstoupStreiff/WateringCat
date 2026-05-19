import io
import logging
import threading
import time

import cv2
import numpy as np
import requests
from dotenv import dotenv_values
from PIL import Image
from ultralytics import YOLO


CANVAS_H, CANVAS_W = (720, 1280)


def draw_mask_on_photo(photo, mask, color=(255, 0, 0), alpha=0.3):
    overlay = photo.copy()
    overlay[mask] = color
    return cv2.addWeighted(photo, 1 - alpha, overlay, alpha, 0)


def load_mask(path):
    try:
        mask = Image.open(path).convert("L")
        return np.array(mask) > 0
    except Exception as e:
        logging.error(f"Error loading mask from {path}: {e}")
        return np.ones((CANVAS_H, CANVAS_W), dtype=bool)


def save_mask(mask, path):
    try:
        Image.fromarray((mask.astype(np.uint8) * 255)).save(path)
    except Exception as e:
        logging.error(f"Error saving mask to {path}: {e}")


def send_discord(webhook_url: str, message: str):
    try:
        requests.post(webhook_url, json={"content": message}, timeout=10)
    except Exception as e:
        logging.error(f"Error sending Discord message: {e}")


def compute_first_mask_iou(cat_mask, activation_mask):
    if cat_mask is None:
        return 0.0
    intersection = np.logical_and(cat_mask, activation_mask)
    return np.sum(intersection) / np.sum(cat_mask) if np.sum(cat_mask) > 0 else 0.0


class Backend:
    def __init__(self):
        self.config = dotenv_values("/.env")
        self.model_name = self.config.get("YOLO_MODEL", "yolov8n.pt")
        self.model = YOLO(self.model_name)

        self.pulling_interval = int(self.config.get("DEFAULT_PULLING_INTERVAL", 5))
        self.activation_mask = load_mask("/app/config/activation_mask.png")
        self.cat_tolerance = float(self.config.get("CAT_TOLERANCE", 0.5))
        self.iou_tolerance = float(self.config.get("IOU_TOLERANCE", 0.3))
        self.pump_duration = int(self.config.get("PUMP_DURATION", 5))

        self.discord_webhook = self.config.get("DISCORD_WEBHOOK", "")
        self.rasp_alive = True

        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def pull_photo(self, force_shape=None):
        try:
            req = requests.get(f"http://{self.config['RASP_ADDRESS']}/shot", timeout=60)
            if req.status_code == 200:
                image = Image.open(io.BytesIO(req.content))
                if force_shape:
                    image = image.resize(force_shape)
                return np.array(image)
            raise Exception(f"HTTP {req.status_code}: {req.text}")
        except Exception as e:
            logging.error(f"Error pulling photo from RASP: {e}")
            raise Exception(f"Failed to pull photo from Raspberry Pi: {e}")

    def activate_pump(self, duration):
        try:
            requests.get(f"http://{self.config['RASP_ADDRESS']}/pump/on?duration={duration}")
        except Exception as e:
            raise Exception(f"Failed to activate pump: {e}")

    def detect_cat(self, photo):
        lab = cv2.cvtColor(photo, cv2.COLOR_RGB2LAB)
        lum, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lum = clahe.apply(lum)
        enhanced = cv2.cvtColor(cv2.merge([lum, a, b]), cv2.COLOR_LAB2RGB)

        results = self.model(enhanced, classes=[15])
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        h, w = photo.shape[:2]

        seg_masks = results[0].masks
        if seg_masks is not None and len(seg_masks) > 0:
            combined = np.zeros((h, w), dtype=bool)
            for m in seg_masks.data.cpu().numpy():
                resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                combined |= resized > 0.5
            mask = combined
        else:
            mask = np.zeros((h, w), dtype=bool)
            for box in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                mask[y1:y2, x1:x2] = True

        best_idx = int(boxes.conf.cpu().numpy().argmax())
        confidence = float(boxes.conf[best_idx].cpu().numpy())
        class_idx = int(boxes.cls[best_idx].cpu().numpy())
        class_name = {14: "bird", 15: "cat", 16: "dog"}.get(class_idx, f"Class {class_idx}")

        return mask, confidence, class_name

    def run(self):
        while True:
            time.sleep(self.pulling_interval)

            try:
                photo = self.pull_photo(force_shape=(CANVAS_W, CANVAS_H))
                if not self.rasp_alive:
                    self.rasp_alive = True
                    logging.info("Raspberry Pi is back online.")
                    if self.discord_webhook:
                        send_discord(self.discord_webhook, "✅ Raspberry Pi is back online!")
            except Exception as e:
                logging.error(f"Error pulling photo: {e}")
                if self.rasp_alive:
                    self.rasp_alive = False
                    logging.warning("Raspberry Pi is unreachable.")
                    if self.discord_webhook:
                        send_discord(self.discord_webhook, "🔴 Raspberry Pi is unreachable!")
                continue

            try:
                detection = self.detect_cat(photo)
            except Exception as e:
                logging.error(f"Error detecting cat: {e}")
                continue

            if detection is None:
                continue

            cat_mask, confidence, class_name = detection
            if cat_mask is not None and confidence >= self.cat_tolerance:
                logging.info(f"{class_name.capitalize()} detected, checking activation mask...")
                iou = compute_first_mask_iou(cat_mask, self.activation_mask)

                if iou > self.iou_tolerance:
                    logging.info(
                        f"{class_name.capitalize()} inside mask (IoU: {iou:.2f}), activating pump for {self.pump_duration}s..."
                    )
                    try:
                        self.activate_pump(self.pump_duration)
                        time.sleep(self.pump_duration)
                    except Exception as e:
                        logging.error(f"Error activating pump: {e}")
                else:
                    logging.info(
                        f"{class_name.capitalize()} not sufficiently inside mask (IoU: {iou:.2f})."
                    )
