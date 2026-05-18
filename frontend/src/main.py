import base64
import io
import logging
import os
import time

import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")
CANVAS_H, CANVAS_W = 720, 1280


def draw_mask_on_photo(photo, mask, color=(255, 0, 0), alpha=0.3):
    result = photo.copy().astype(np.float32)
    result[mask] = (
        result[mask] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    )
    return result.clip(0, 255).astype(np.uint8)


def photo_to_bytes(photo: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(photo).save(buf, format="JPEG")
    return buf.getvalue()


def bytes_to_photo(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def fetch_activation_mask() -> np.ndarray | None:
    try:
        resp = requests.get(f"{BACKEND_URL}/mask", timeout=10)
        resp.raise_for_status()
        return np.array(Image.open(io.BytesIO(resp.content)).convert("L")) > 0
    except Exception as e:
        st.error(f"Error loading activation mask: {e}")
        return None


def main():
    st.set_page_config(page_title="Watering Cat", page_icon="💧", layout="wide")

    try:
        status = requests.get(f"{BACKEND_URL}/status", timeout=5).json()
    except Exception as e:
        st.error(f"Cannot reach backend at {BACKEND_URL}: {e}")
        return

    with st.sidebar:
        tab = st.segmented_control(
            "Menu", ["Monitor", "Mask Editor"], default="Monitor", key="main_tab"
        )
        st.divider()
        st.space()

    if tab == "Monitor":
        cols = st.columns(5)
        cols[0].metric("Cat Tolerance", f"{status['cat_tolerance']:.2%}")
        cols[1].metric("IoU Tolerance", f"{status['iou_tolerance']:.2%}")
        cols[2].metric("Pump Duration", f"{status['pump_duration']}s")
        cols[3].metric("Pulling Interval", f"{status['pulling_interval']}s")
        cols[4].metric("YOLO Model", status["model_name"])

        with st.sidebar:
            button_take_photo = st.button("📸 Take a Photo", use_container_width=True)

            def change_button_upload_photo():
                st.session_state["change_button_upload_photo"] = True

            button_upload_photo = st.file_uploader(
                "📤 Upload a photo",
                type=["jpg", "jpeg", "png"],
                on_change=change_button_upload_photo,
            )
            button_water_plant = st.button("🚿 Water Plants", use_container_width=True)

        st.divider()
        cols = st.columns(2)
        photo = None

        with cols[0]:
            if button_take_photo:
                try:
                    resp = requests.get(f"{BACKEND_URL}/photo", timeout=60)
                    resp.raise_for_status()
                    photo = bytes_to_photo(resp.content)
                    st.session_state["last_photo"] = photo
                    st.toast("Photo updated!")
                except Exception as e:
                    st.error(f"Error taking photo: {e}")
            if button_upload_photo and st.session_state.get(
                "change_button_upload_photo", False
            ):
                st.session_state["change_button_upload_photo"] = False
                try:
                    image = Image.open(button_upload_photo).convert("RGB")
                    photo = np.array(image.resize((CANVAS_W, CANVAS_H)))
                    st.session_state["last_photo"] = photo
                    st.toast("Photo uploaded!")
                except Exception as e:
                    st.error(f"Error uploading photo: {e}")
            photo = st.session_state.get("last_photo")
            if photo is not None:
                st.image(photo, caption="Latest Photo from Raspberry Pi")

        with cols[1]:
            if button_water_plant:
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/pump?duration={status['pump_duration']}"
                    )
                    resp.raise_for_status()
                    st.toast("Pump activated!")
                except Exception as e:
                    st.error(f"Error activating pump: {e}")
            activation_mask = fetch_activation_mask()
            if activation_mask is not None:
                st.image(
                    activation_mask.astype(np.uint8) * 255, caption="Activation Mask"
                )

        if photo is not None and activation_mask is not None:
            with cols[0]:
                try:
                    with st.spinner("Detecting cat..."):
                        start = time.time()
                        resp = requests.post(
                            f"{BACKEND_URL}/detect",
                            files={
                                "file": (
                                    "photo.jpg",
                                    photo_to_bytes(photo),
                                    "image/jpeg",
                                )
                            },
                        )
                        resp.raise_for_status()
                        result = resp.json()
                        end = time.time()

                    if result["detected"]:
                        cat_mask_bytes = base64.b64decode(result["cat_mask_png"])
                        cat_mask = (
                            np.array(
                                Image.open(io.BytesIO(cat_mask_bytes)).convert("L")
                            )
                            > 0
                        )
                        if cat_mask.shape != photo.shape[:2]:
                            cat_mask = (
                                np.array(
                                    Image.fromarray(
                                        cat_mask.astype(np.uint8) * 255
                                    ).resize(
                                        (photo.shape[1], photo.shape[0]), Image.NEAREST
                                    )
                                )
                                > 0
                            )
                        full_img = draw_mask_on_photo(
                            photo, cat_mask, color=(0, 255, 0), alpha=0.6
                        )
                        full_img = draw_mask_on_photo(
                            full_img, activation_mask, color=(255, 0, 0), alpha=0.3
                        )
                        st.image(
                            full_img, caption="Cat (green) + Activation Mask (red)"
                        )
                        label = result["class_name"].capitalize()
                        st.success(
                            f"{label} detected: {result['confidence']:.2%} confidence in {end - start:.2f}s"
                        )
                        st.info(
                            f"{label} is {result['iou']:.2%} inside activation mask"
                            f" (Tolerance: {status['iou_tolerance']:.2%})"
                        )
                    else:
                        st.warning("No cat detected in photo.")
                except Exception as e:
                    st.error(f"Error detecting cat: {e}")
            with cols[1]:
                st.image(
                    draw_mask_on_photo(photo, activation_mask, color=(255, 0, 0)),
                    caption="Activation Mask Overlay",
                )

    if tab == "Mask Editor":
        st.subheader("Draw Activation Mask")
        st.caption(
            "Paint the area where the cat should trigger the pump. White = active zone."
        )

        bg_arr = st.session_state.get("last_photo")
        bg_image = (
            np.array(Image.fromarray(bg_arr).resize((CANVAS_W, CANVAS_H)))
            if bg_arr is not None
            else np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        )

        with st.sidebar:
            drawing_mode = st.radio(
                "Mode", ["freedraw", "rect", "erase"], horizontal=False
            )
            brush_size = (
                st.slider("Brush size", 5, 80, 20) if drawing_mode == "freedraw" else 0
            )

        is_erase = drawing_mode == "erase"
        color = "rgba(0,0,0,1)" if is_erase else "rgba(255,0,0,0.3)"
        canvas_drawing_mode = "freedraw" if is_erase else drawing_mode

        canvas_result = st_canvas(
            fill_color=color,
            stroke_width=brush_size,
            stroke_color=color,
            background_image=Image.fromarray(bg_image),
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
            drawn = canvas_result.image_data[:, :, :3]
            new_mask = np.mean(drawn, axis=2) > 10

            existing_mask = fetch_activation_mask()
            act_h, act_w = (
                existing_mask.shape
                if existing_mask is not None
                else (CANVAS_H, CANVAS_W)
            )

            new_mask_full = (
                np.array(
                    Image.fromarray((new_mask.astype(np.uint8) * 255)).resize(
                        (act_w, act_h), Image.NEAREST
                    )
                )
                > 0
            )

            st.image(
                new_mask_full.astype(np.uint8) * 255,
                caption="New Activation Mask Preview",
            )

            if st.button("Save Mask", type="primary", use_container_width=True):
                buf = io.BytesIO()
                Image.fromarray((new_mask_full.astype(np.uint8) * 255)).save(
                    buf, format="PNG"
                )
                buf.seek(0)
                try:
                    resp = requests.put(
                        f"{BACKEND_URL}/mask",
                        files={"file": ("mask.png", buf.getvalue(), "image/png")},
                    )
                    resp.raise_for_status()
                    st.success("Mask saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving mask: {e}")


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
main()
