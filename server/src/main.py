import logging
import time

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from backend import (
    CANVAS_H,
    CANVAS_W,
    Backend,
    compute_first_mask_iou,
    draw_mask_on_photo,
    save_mask,
)


def main():
    st.set_page_config(page_title="Watering Cat", page_icon="🐱💧", layout="wide")
    tab = st.segmented_control(["Monitor", "Mask Editor"], key="main_tab")

    if tab == "Monitor":
        photo = None
        backend = st.session_state.backend
        cat_tolerance = backend.cat_tolerance
        iou_tolerance = backend.iou_tolerance
        pump_duration = backend.pump_duration
        pulling_interval = backend.pulling_interval

        cols = st.columns(5)
        cols[0].metric("Cat Tolerance", f"{cat_tolerance:.2%}")
        cols[1].metric("IoU Tolerance", f"{iou_tolerance:.2%}")
        cols[2].metric("Pump Duration", f"{pump_duration}s")
        cols[3].metric("Pulling Interval", f"{pulling_interval}s")
        cols[4].metric("YOLO Model", backend.model_name)

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

        with cols[0]:
            if button_take_photo:
                try:
                    photo = backend.pull_photo(force_shape=(CANVAS_W, CANVAS_H))
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
                    st.toast("Photo uploaded and set as last photo!")
                except Exception as e:
                    st.error(f"Error uploading photo: {e}")
            photo = st.session_state.get("last_photo")
            if photo is not None:
                st.image(photo, caption="Latest Photo from Raspberry Pi")

        with cols[1]:
            if button_water_plant:
                try:
                    backend.activate_pump(backend.pump_duration)
                    st.toast("Pump activated!")
                except Exception as e:
                    st.error(f"Error activating pump: {e}")
            activation_mask_img = backend.activation_mask.astype(np.uint8) * 255
            st.image(activation_mask_img, caption="Activation Mask")

        if photo is not None:
            with cols[0]:
                try:
                    with st.spinner("Detecting cat..."):
                        start = time.time()
                        detection = backend.detect_cat(photo)
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
                            backend.activation_mask,
                            color=(255, 0, 0),
                            alpha=0.3,
                        )
                        st.image(
                            full_img, caption="Cat (green) + Activation Mask (red)"
                        )
                        iou = compute_first_mask_iou(cat_mask, backend.activation_mask)
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
                        photo, backend.activation_mask, color=(255, 0, 0)
                    ),
                    caption="Activation Mask Overlay",
                )

    if tab == "Mask Editor":
        bg_arr = st.session_state.get("last_photo")
        if bg_arr is not None:
            bg_image = np.array(Image.fromarray(bg_arr).resize((CANVAS_W, CANVAS_H)))
        else:
            bg_image = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        with st.sidebar:
            drawing_mode = st.radio(
                "Mode", ["freedraw", "rect", "erase"], horizontal=True
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
            gray = np.mean(drawn, axis=2)
            new_mask = gray > 10
            act_h, act_w = st.session_state.backend.activation_mask.shape
            new_mask_img = Image.fromarray((new_mask.astype(np.uint8) * 255)).resize(
                (act_w, act_h), Image.NEAREST
            )
            new_mask_full = np.array(new_mask_img) > 0

            st.image(
                new_mask_full.astype(np.uint8) * 255,
                caption="New Activation Mask Preview",
            )

        with st.sidebar:
            if st.button(
                "Save Mask",
                type="primary",
                use_container_width=True,
                disabled=canvas_result.image_data is None,
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
