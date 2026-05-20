import base64
import datetime
import io
import logging
import os
import time

import numpy as np
import pandas as pd
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


def build_timeline_image(
    events_json, window_hours, segment_minutes, width=1200, height=60
):
    now = datetime.datetime.now()
    start_time = now - datetime.timedelta(hours=window_hours)
    total_secs = window_hours * 3600
    n_segments = max(1, int(total_secs / (segment_minutes * 60)))

    GRAY = np.array([120, 120, 120], dtype=np.uint8)
    RED = np.array([210, 60, 60], dtype=np.uint8)
    GREEN = np.array([60, 190, 80], dtype=np.uint8)
    BLUE = np.array([60, 120, 220], dtype=np.uint8)
    PURPLE = np.array([160, 80, 200], dtype=np.uint8)

    parsed = []
    for e in events_json:
        try:
            parsed.append((datetime.datetime.fromisoformat(e["timestamp"]), e))
        except Exception:
            continue
    parsed.sort(key=lambda x: x[0])

    initial_alive = True
    for ts, e in parsed:
        if ts >= start_time:
            break
        if e["type"] == "rasp_down":
            initial_alive = False
        elif e["type"] == "rasp_up":
            initial_alive = True

    in_window = [(ts, e) for ts, e in parsed if start_time <= ts < now]

    segment_colors = []
    rasp_alive = initial_alive
    ptr = 0
    seg_secs = segment_minutes * 60

    for i in range(n_segments):
        seg_start = start_time + datetime.timedelta(seconds=i * seg_secs)
        seg_end = seg_start + datetime.timedelta(seconds=seg_secs)
        has_cat = has_pump = False
        was_offline = not rasp_alive

        has_cpu_heat = False
        while ptr < len(in_window):
            ts, e = in_window[ptr]
            if ts >= seg_end:
                break
            ptr += 1
            if e["type"] == "rasp_down":
                was_offline = True
                rasp_alive = False
            elif e["type"] == "rasp_up":
                rasp_alive = True
            elif e["type"] == "cat_detected":
                has_cat = True
            elif e["type"] == "pump_activated":
                has_pump = True
            elif e["type"] == "cpu_heat_warning":
                has_cpu_heat = True

        if has_cat:
            segment_colors.append(GREEN)
        elif has_pump:
            segment_colors.append(BLUE)
        elif was_offline:
            segment_colors.append(GRAY)
        elif has_cpu_heat:
            segment_colors.append(PURPLE)
        else:
            segment_colors.append(RED)

    pixels = np.zeros((width, 3), dtype=np.uint8)
    for px in range(width):
        pixels[px] = segment_colors[min(int(px * n_segments / width), n_segments - 1)]

    if n_segments <= width // 3:
        for i in range(1, n_segments):
            px = int(i * width / n_segments)
            if px < width:
                pixels[px] = [20, 20, 20]

    return np.repeat(pixels[np.newaxis, :, :], height, axis=0)


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
            "Menu",
            ["Monitor", "History", "Mask Editor"],
            default="Monitor",
            required=True,
            key="main_tab",
        )
        st.divider()
        st.space()

    if tab == "Monitor":
        cols = st.columns(6)
        cols[0].metric("Cat Tolerance", f"{status['cat_tolerance']:.2%}")
        cols[1].metric("IoU Tolerance", f"{status['iou_tolerance']:.2%}")
        cols[2].metric("Pump Duration", f"{status['pump_duration']}s")
        cols[3].metric("Pulling Interval", f"{status['pulling_interval']}s")
        cols[4].metric("YOLO Model", status["model_name"])
        cpu_temp = status.get("cpu_temperature")
        cols[5].metric(
            "CPU Temp", f"{cpu_temp:.1f}°C" if cpu_temp is not None else "N/A"
        )

        with st.sidebar:
            auto_refresh = st.toggle("🔄 Auto Refresh (2s)", value=False)
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
            if button_take_photo or auto_refresh:
                try:
                    resp = requests.get(f"{BACKEND_URL}/photo", timeout=60)
                    resp.raise_for_status()
                    photo = bytes_to_photo(resp.content)
                    st.session_state["last_photo"] = photo
                    if button_take_photo:
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
                        st.warning(f"No cat detected in photo in {end - start:.2f}s")
                except Exception as e:
                    st.error(f"Error detecting cat: {e}")
            with cols[1]:
                st.image(
                    draw_mask_on_photo(photo, activation_mask, color=(255, 0, 0)),
                    caption="Activation Mask Overlay",
                )

        if auto_refresh:
            time.sleep(2)
            st.rerun()

    if tab == "History":
        try:
            resp = requests.get(f"{BACKEND_URL}/events?limit=1000", timeout=10)
            resp.raise_for_status()
            all_events = resp.json()
        except Exception as e:
            st.error(f"Cannot fetch events: {e}")
            all_events = []

        try:
            resp = requests.get(f"{BACKEND_URL}/cpu-temperature/history?limit=5760", timeout=10)
            resp.raise_for_status()
            all_cpu_temps = resp.json()
        except Exception:
            all_cpu_temps = []

        window_label = st.segmented_control(
            "Time Window",
            ["1h", "6h", "12h", "24h", "48h", "168h"],
            default="24h",
            required=True,
            key="history_window",
        )
        segment_label = st.segmented_control(
            "Segment",
            ["5m", "15m", "1h"],
            default="15m",
            required=True,
            key="history_segment",
        )
        window_hours = int(window_label.rstrip("h"))
        segment_minutes = {"5m": 5, "15m": 15, "1h": 60}[segment_label]

        timeline_img = build_timeline_image(all_events, window_hours, segment_minutes)
        st.image(timeline_img, use_container_width=True)

        start_label = (
            datetime.datetime.now() - datetime.timedelta(hours=window_hours)
        ).strftime("%m/%d %H:%M")
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:gray;margin-top:-8px">'
            f"<span>{start_label}</span><span>now</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="display:flex;gap:1.5rem;font-size:0.8rem;margin-bottom:8px">'
            '<span><span style="color:#3c78dc">■</span> Pump activated</span>'
            '<span><span style="color:#3cbe50">■</span> Cat detected</span>'
            '<span><span style="color:#d23c3c">■</span> No detection</span>'
            '<span><span style="color:#787878">■</span> Offline</span>'
            '<span><span style="color:#a050c8">■</span> CPU heat warning</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(hours=window_hours)
        events = [
            e
            for e in all_events
            if datetime.datetime.fromisoformat(e["timestamp"]) >= cutoff
        ]

        if not events:
            st.info("No events in this time window.")
        else:
            df = pd.DataFrame(events)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            cat_df = df[df["type"] == "cat_detected"]
            pump_count = df[df["type"] == "pump_activated"]

            cols = st.columns(4)
            cols[0].metric("Cat Detections", len(cat_df))
            cols[1].metric("Pump Activations", len(pump_count))
            cols[2].metric("Total Events", len(df))
            if len(cat_df) > 0:
                cols[3].metric(
                    "Last Detection",
                    cat_df.iloc[0]["timestamp"].strftime("%m/%d %H:%M"),
                )

            st.divider()

            if len(cat_df) > 0:
                st.subheader("Detections per Hour")
                hourly = (
                    cat_df.set_index("timestamp")
                    .resample("h")
                    .size()
                    .rename("detections")
                    .reset_index()
                    .set_index("timestamp")
                )
                st.bar_chart(hourly)

            if all_cpu_temps:
                cpu_df = pd.DataFrame(all_cpu_temps)
                cpu_df["timestamp"] = pd.to_datetime(cpu_df["timestamp"])
                cpu_df = cpu_df[cpu_df["timestamp"] >= cutoff].set_index("timestamp")
                if not cpu_df.empty:
                    st.subheader("CPU Temperature (°C)")
                    st.line_chart(cpu_df["temperature"])

            st.subheader("Event Log")
            type_icons = {
                "cat_detected": "🐱",
                "rasp_down": "🔴",
                "rasp_up": "✅",
                "cpu_heat_warning": "🌡️",
                "pump_activated": "🚿",
            }
            display = df.copy()
            display[""] = display["type"].map(lambda t: type_icons.get(t, "•"))
            display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            cols_order = ["", "timestamp", "type"]
            for col in ["class_name", "confidence", "iou", "in_zone", "pump_activated"]:
                if col in display.columns:
                    cols_order.append(col)
            st.dataframe(display[cols_order], use_container_width=True, hide_index=True)

        st.divider()
        if st.button("Clear History", type="secondary", use_container_width=True):
            try:
                requests.delete(f"{BACKEND_URL}/events", timeout=10)
                st.toast("History cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing history: {e}")

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
