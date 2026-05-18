import io
import logging
import threading
import time

import cv2
import numpy as np
import requests
import streamlit as st
from dotenv import dotenv_values
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO


CANVAS_H, CANVAS_W = (720, 1280)


def draw_mask_on_photo(photo, mask, color=(255, 0, 0), alpha=0.3):
    """
    Draws the activation mask on the photo for visualization.
    Args:
        photo (np.ndarray): The original photo as a NumPy array.
        mask (np.ndarray): The activation mask as a boolean NumPy array.
    Returns:
        np.ndarray: The photo with the activation mask drawn on it.
    """
    overlay = photo.copy()
    overlay[mask] = color  # Highlight the masked area in the specified color
    blended = cv2.addWeighted(
        photo, 1 - alpha, overlay, alpha, 0
    )  # Blend the original photo with the overlay
    return blended


def load_mask(path):
    """
    Loads the activation mask from a file.
    Args:
        path (str): The file path to the activation mask image.
    Returns:
        np.ndarray: The activation mask as a boolean NumPy array, where True indicates the area to activate the pump.
    Raises:
        Exception: If there is an error loading the mask from the file, a default mask activating the
    """
    try:
        mask = Image.open(path).convert(
            "L"
        )  # Load the mask image and convert to grayscale
        return (
            np.array(mask) > 0
        )  # Convert to boolean array where True indicates the area to activate the pump
    except Exception as e:
        logging.error(f"Error loading mask from {path}: {e}")
        return np.ones(
            (CANVAS_H, CANVAS_W), dtype=bool
        )  # Default to activating the pump for the entire area if there's an error loading the mask


def save_mask(mask, path):
    """Saves the activation mask to a file.
    Args:
        mask (np.ndarray): The activation mask as a boolean NumPy array.
        path (str): The file path where the mask should be saved.
    """
    try:
        mask_image = Image.fromarray(
            (mask.astype(np.uint8) * 255)
        )  # Convert boolean array back to image format
        mask_image.save(path)  # Save the mask image to the specified path
    except Exception as e:
        logging.error(f"Error saving mask to {path}: {e}")


def compute_first_mask_iou(cat_mask, activation_mask):
    """
    Computes the percentage of the cat mask that is inside the activation mask.
    Args:
        cat_mask (np.ndarray): A boolean mask of the same size as the input photo, where True indicates the presence of a cat.
        activation_mask (np.ndarray): A boolean mask of the same size as the input photo, where True indicates the area to activate the pump.
    Returns:
        float: The percentage of the cat mask that is inside the activation mask. Returns 0 if no cats are detected.
    """
    if cat_mask is None:
        return 0.0
    intersection = np.logical_and(cat_mask, activation_mask)
    return np.sum(intersection) / np.sum(cat_mask) if np.sum(cat_mask) > 0 else 0.0


class Backend:
    def __init__(self):
        self.config = dotenv_values("/.env")  # Load configuration from .env file
        self.model_name = self.config.get("YOLO_MODEL", "yolov8n.pt")
        self.model = YOLO(
            self.model_name
        )  # Yolo model, pretrained on the COCO dataset, which includes cats as one of the classes.

        self.pulling_interval = int(self.config.get("DEFAULT_PULLING_INTERVAL", 5))
        self.activation_mask = load_mask(
            "/app/config/activation_mask.png"
        )  # Mask of the area to activate the pump if the cat is inside.
        self.cat_tolerance = float(
            self.config.get("CAT_TOLERANCE", 0.5)
        )  # Confidence threshold for cat detection.
        self.iou_tolerance = float(
            self.config.get("IOU_TOLERANCE", 0.3)
        )  # Percentage of the cat that needs to be inside the activation mask.
        self.pump_duration = int(
            self.config.get("PUMP_DURATION", 5)
        )  # Duration for which the pump should be activated.

        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def pull_photo(self, force_shape=None):
        """
        Pulls a photo from the Raspberry Pi.
        Returns:
            np.ndarray: The photo as a NumPy array.
        Raises:
            Exception: If there is an error pulling the photo from the Raspberry Pi or loading the fallback
        """
        # Try to pull photo from Raspberry Pi
        try:
            req = requests.get(f"http://{self.config['RASP_ADDRESS']}/shot", timeout=60)
            if req.status_code == 200:
                image = Image.open(io.BytesIO(req.content))
                if force_shape:
                    image = image.resize(force_shape)
                return np.array(image)
            logging.error(
                f"Failed to pull photo from RASP: {req.status_code} - {req.text}"
            )
            raise Exception(
                f"Failed to pull photo from Raspberry Pi: {req.status_code} - {req.text}"
            )
        except Exception as e:
            logging.error(f"Error pulling photo from RASP: {e}")
            raise Exception(f"Failed to pull photo from Raspberry Pi: {e}")

    def activate_pump(self, duration):
        """Activates the pump for a specified duration by sending a request to the Raspberry Pi.
        Args:
            duration (int): The duration in seconds for which the pump should be activated.
        Raises:
            Exception: If there is an error sending the request to activate the pump.
        """
        try:
            requests.get(
                f"http://{self.config['RASP_ADDRESS']}/pump/on?duration={duration}"
            )
        except Exception as e:
            raise Exception(f"Failed to activate pump: {e}")

    def detect_cat(self, photo):
        """
        Detects cats in the given photo using the YOLO model.
        Args:
            photo (np.ndarray): The input photo as a NumPy array (RGB).
        Returns:
            tuple or None: (mask, confidence, class_name) where mask is a boolean array,
                           or None if no cats are detected.
        """
        # Enhance contrast using CLAHE on the L channel (works in any lighting)
        lab = cv2.cvtColor(photo, cv2.COLOR_RGB2LAB)
        lum, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lum = clahe.apply(lum)
        enhanced = cv2.cvtColor(cv2.merge([lum, a, b]), cv2.COLOR_LAB2RGB)

        results = self.model(enhanced, classes=[15])  # Class 15 = cat only
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        h, w = photo.shape[:2]

        # Use segmentation masks when available (seg model), fall back to boxes
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

        # Pick confidence and class from the highest-confidence detection
        best_idx = int(boxes.conf.cpu().numpy().argmax())
        confidence = float(boxes.conf[best_idx].cpu().numpy())
        class_idx = int(boxes.cls[best_idx].cpu().numpy())
        class_name = {14: "bird", 15: "cat", 16: "dog"}.get(class_idx, f"Class {class_idx}")

        return mask, confidence, class_name

    def run(self):
        """
        Main loop that continuously pulls photos from the Raspberry Pi, detects cats, checks if they are within the activation mask, and activates the pump if necessary.
        """
        while True:
            time.sleep(self.pulling_interval)

            # Pull photo from Raspberry Pi
            try:
                photo = self.pull_photo(force_shape=(CANVAS_W, CANVAS_H))
            except Exception as e:
                logging.error(f"Error pulling photo: {e}")
                continue

            # Detect cat in the photo
            try:
                detection = self.detect_cat(photo)
            except Exception as e:
                logging.error(f"Error detecting cat: {e}")
                continue

            # Check if any cats were detected
            if detection is None:
                continue
            cat_mask, confidence, class_name = detection
            if cat_mask is not None and confidence >= self.cat_tolerance:
                logging.info(
                    f"{class_name.capitalize()} detected in photo, checking activation mask..."
                )
                iou = compute_first_mask_iou(cat_mask, self.activation_mask)

                if iou > self.iou_tolerance:
                    logging.info(
                        f"{class_name.capitalize()} is inside activation mask (IoU: {iou:.2f}), activating pump for {self.pump_duration} seconds..."
                    )
                    try:
                        self.activate_pump(self.pump_duration)
                        time.sleep(self.pump_duration)
                    except Exception as e:
                        logging.error(f"Error activating pump: {e}")
                else:
                    logging.info(
                        f"{class_name.capitalize()} is not sufficiently inside activation mask (IoU: {iou:.2f}), not activating pump."
                    )


def main():
    st.set_page_config(page_title="Watering Cat", page_icon="🐱💧", layout="wide")

    tab_monitor, tab_mask = st.tabs(["Monitor", "Mask Editor"])

    with tab_monitor:
        photo = None
        cat_tolerance = st.session_state.backend.cat_tolerance
        iou_tolerance = st.session_state.backend.iou_tolerance
        pump_duration = st.session_state.backend.pump_duration
        pulling_interval = st.session_state.backend.pulling_interval

        cols = st.columns(5)
        cols[0].metric("Cat Tolerance", f"{cat_tolerance:.2%}")
        cols[1].metric("IoU Tolerance", f"{iou_tolerance:.2%}")
        cols[2].metric("Pump Duration", f"{pump_duration}s")
        cols[3].metric("Pulling Interval", f"{pulling_interval}s")
        cols[4].metric("YOLO Model", st.session_state.backend.model_name)
        cols = st.columns(2)

        with cols[0]:
            button_take_photo = st.button("📸 Take a Photo", use_container_width=True)
            button_upload_photo = st.file_uploader(
                "📤 Upload a photo", type=["jpg", "jpeg", "png"]
            )
        with cols[1]:
            button_water_plant = st.button("🚿 Water Plants", use_container_width=True)

        st.divider()
        cols = st.columns(2)

        with cols[0]:
            if button_take_photo:
                try:
                    photo = st.session_state.backend.pull_photo(
                        force_shape=(CANVAS_W, CANVAS_H)
                    )
                    st.session_state["last_photo"] = photo
                    st.toast("Photo updated!")
                except Exception as e:
                    st.error(f"Error taking photo: {e}")
            if button_upload_photo:
                try:
                    image = Image.open(button_upload_photo).convert("RGB")
                    photo = np.array(image.resize((CANVAS_W, CANVAS_H)))
                    st.session_state["last_photo"] = photo
                    st.toast("Photo uploaded and set as last photo!")
                except Exception as e:
                    st.error(f"Error uploading photo: {e}")
            photo = st.session_state.get("last_photo")
            if photo is not None:
                st.image(photo, caption="Latest Photo from Raspberry Pi")

        with cols[1]:
            if button_water_plant:
                try:
                    st.session_state.backend.activate_pump(
                        st.session_state.backend.pump_duration
                    )
                    st.toast("Pump activated!")
                except Exception as e:
                    st.error(f"Error activating pump: {e}")
            activation_mask_img = (
                st.session_state.backend.activation_mask.astype(np.uint8) * 255
            )
            st.image(activation_mask_img, caption="Activation Mask")

        if photo is not None:
            with cols[0]:
                try:
                    with st.spinner("Detecting cat..."):
                        start = time.time()
                        detection = st.session_state.backend.detect_cat(photo)
                        end = time.time()
                    cat_mask, confidence, class_name = (
                        detection if detection is not None else (None, 0.0, "")
                    )
                    if cat_mask is not None:
                        full_img = draw_mask_on_photo(
                            photo, cat_mask, color=(0, 255, 0), alpha=0.6
                        )
                        full_img = draw_mask_on_photo(
                            full_img,
                            st.session_state.backend.activation_mask,
                            color=(255, 0, 0),
                            alpha=0.3,
                        )
                        st.image(
                            full_img, caption="Cat (green) + Activation Mask (red)"
                        )
                        iou = compute_first_mask_iou(
                            cat_mask, st.session_state.backend.activation_mask
                        )
                        st.success(
                            f"{class_name.capitalize()} detected: {confidence:.2%} confidence in {end - start:.2f}s"
                        )
                        st.info(
                            f"{class_name.capitalize()} is {iou:.2%} inside activation mask (Tolerance: {iou_tolerance:.2%})"
                        )
                    else:
                        st.warning("No cat detected in photo.")
                except Exception as e:
                    st.error(f"Error detecting cat: {e}")
            with cols[1]:
                st.image(
                    draw_mask_on_photo(
                        photo,
                        st.session_state.backend.activation_mask,
                        color=(255, 0, 0),
                    ),
                    caption="Activation Mask Overlay",
                )

    with tab_mask:
        st.subheader("Draw Activation Mask")
        st.caption(
            "Paint the area where the cat should trigger the pump. White = active zone."
        )
        bg_arr = st.session_state.get("last_photo")
        if bg_arr is not None:
            bg_image = np.array(Image.fromarray(bg_arr).resize((CANVAS_W, CANVAS_H)))
        else:
            bg_image = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        drawing_mode = st.radio("Mode", ["freedraw", "rect", "erase"], horizontal=True)
        if drawing_mode == "freedraw":
            brush_size = st.slider("Brush size", 5, 80, 20)
        else:
            brush_size = 0

        is_erase = drawing_mode == "erase"
        color = "rgba(0,0,0,1)" if is_erase else "rgba(255,0,0,0.3)"
        canvas_drawing_mode = "freedraw" if is_erase else drawing_mode

        pil_image = Image.fromarray(bg_image)

        canvas_result = st_canvas(
            fill_color=color,
            stroke_width=brush_size,
            stroke_color=color,
            background_image=pil_image,
            background_color="#000000",
            update_streamlit=True,
            width=CANVAS_W,
            height=CANVAS_H,
            drawing_mode=canvas_drawing_mode,
            key="mask_canvas",
        )

        if canvas_result.image_data is None:
            st.info("Draw on the canvas to define the activation mask.")
        else:
            drawn = canvas_result.image_data[:, :, :3]  # RGB from canvas
            gray = np.mean(drawn, axis=2)
            new_mask = gray > 10
            act_h, act_w = st.session_state.backend.activation_mask.shape
            new_mask_img = Image.fromarray((new_mask.astype(np.uint8) * 255)).resize(
                (act_w, act_h), Image.NEAREST
            )
            new_mask_full = np.array(new_mask_img) > 0

            st.image(
                new_mask_full.astype(np.uint8) * 255, caption="New Activation Mask Preview"
            )

            if st.button(
                "Save Mask",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.backend.activation_mask = new_mask_full
                save_mask(new_mask_full, "/app/config/activation_mask.png")
                st.success("Mask saved successfully!")
                st.rerun()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

with st.spinner("Initializing..."):
    if "backend" not in st.session_state:
        logging.info("Initializing backend...")
        st.session_state.backend = Backend()
        logging.info("Backend initialized and thread started")
main()
