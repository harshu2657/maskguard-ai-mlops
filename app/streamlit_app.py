
import io
import tempfile
from pathlib import Path

import av
import requests
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from video_utils import load_detection_model, process_video


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Face Mask Detection MLOps",
    page_icon="😷",
    layout="wide",
)

st.title("😷 Face Mask Detection MLOps")
st.write(
    "Detect mask, bad mask, and no mask using your trained YOLOv8 model."
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_detection_model()


# --------------------------------------------------
# DISPLAY DETECTION COUNTS
# --------------------------------------------------

def show_detection_counts(counts, total=None):
    col1, col2, col3, col4 = st.columns(4)

    mask_count = counts.get("mask", 0)
    no_mask_count = counts.get("no mask", 0)
    bad_mask_count = counts.get("bad mask", 0)

    col1.metric("Mask", mask_count)
    col2.metric("No Mask", no_mask_count)
    col3.metric("Bad Mask", bad_mask_count)

    if total is None:
        total = mask_count + no_mask_count + bad_mask_count

    col4.metric("Total Detections", total)


# --------------------------------------------------
# IMAGE PREDICTION
# --------------------------------------------------

def image_page():
    st.header("Image Detection")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        key="image_confidence",
    )

    if uploaded and st.button("Detect Masks", key="image_detect"):
        try:
            with st.spinner("Detecting masks..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    files={
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type or "image/jpeg",
                        )
                    },
                    params={"conf": confidence},
                    timeout=180,
                )

                if not response.ok:
                    st.error(
                        f"API error ({response.status_code}): "
                        f"{response.text}"
                    )
                    return

                result = response.json()
                detections = result.get("detections", [])

                image = Image.open(
                    io.BytesIO(uploaded.getvalue())
                ).convert("RGB")

                draw = ImageDraw.Draw(image)

                colors = {
                    "mask": "#16A34A",
                    "no mask": "#DC2626",
                    "bad mask": "#F59E0B",
                }

                counts = {
                    "mask": 0,
                    "no mask": 0,
                    "bad mask": 0,
                }

                rows = []

                for detection in detections:
                    name = detection.get("class_name", "unknown")
                    score = float(detection.get("confidence", 0))
                    box = detection.get("bounding_box", {})

                    if name in counts:
                        counts[name] += 1

                    x1 = int(box.get("x1", 0))
                    y1 = int(box.get("y1", 0))
                    x2 = int(box.get("x2", 0))
                    y2 = int(box.get("y2", 0))

                    color = colors.get(name, "#2563EB")

                    x1 = max(0, min(x1, image.width - 1))
                    x2 = max(0, min(x2, image.width - 1))
                    y1 = max(0, min(y1, image.height - 1))
                    y2 = max(0, min(y2, image.height - 1))

                    draw.rectangle(
                        [x1, y1, x2, y2],
                        outline=color,
                        width=4,
                    )

                    label = f"{name}: {score:.0%}"
                    text_box = draw.textbbox((x1, y1), label)

                    text_width = text_box[2] - text_box[0]
                    text_height = text_box[3] - text_box[1]
                    label_y = max(0, y1 - text_height - 8)

                    draw.rectangle(
                        [
                            x1,
                            label_y,
                            min(image.width - 1, x1 + text_width + 8),
                            min(
                                image.height - 1,
                                label_y + text_height + 8,
                            ),
                        ],
                        fill=color,
                    )

                    draw.text(
                        (x1 + 4, label_y + 4),
                        label,
                        fill="white",
                    )

                    rows.append({
                        "Class": name,
                        "Confidence": f"{score:.1%}",
                    })

            st.success("Prediction completed!")

            show_detection_counts(
                counts,
                result.get("detection_count", len(detections)),
            )

            original_col, result_col = st.columns(2)

            with original_col:
                st.subheader("Original Image")
                st.image(
                    uploaded.getvalue(),
                    use_container_width=True,
                )

            with result_col:
                st.subheader("Detected Image")
                st.image(
                    image,
                    use_container_width=True,
                )

                output_buffer = io.BytesIO()
                image.save(output_buffer, format="PNG")

                st.download_button(
                    "Download Detected Image",
                    data=output_buffer.getvalue(),
                    file_name="face_mask_detection.png",
                    mime="image/png",
                )

            st.subheader("Detection Details")

            if rows:
                st.dataframe(
                    rows,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No objects were detected.")

        except requests.RequestException:
            st.error(
                "Cannot connect to FastAPI. "
                "Make sure the API is running on port 8000."
            )

        except Exception as error:
            st.error(f"Image prediction failed: {error}")


# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------

def video_page():
    st.header("Video Detection")

    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        key="video_confidence",
    )

    if uploaded:
        st.subheader("Original Video")
        st.video(uploaded.getvalue())

        if st.button("Process Video", key="process_video"):
            input_path = None
            output_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded.name).suffix or ".mp4",
                ) as temp:
                    temp.write(uploaded.getvalue())
                    input_path = temp.name

                with st.spinner(
                    "Processing video on CPU. "
                    "Long videos may take some time..."
                ):
                    model = get_model()

                    (
                        output_path,
                        counts,
                        processed,
                        total,
                    ) = process_video(
                        input_path,
                        model,
                        confidence=confidence,
                    )

                st.success("Video processing completed!")

                st.write(
                    f"**Frames processed:** {processed} / {total}"
                )

                st.subheader("Detection Summary")
                show_detection_counts(counts)

                st.caption(
                    "Counts include detections across all processed frames. "
                    "The same person may be counted in multiple frames."
                )

                st.subheader("Annotated Video")
                st.video(output_path)

                with open(output_path, "rb") as file:
                    video_bytes = file.read()

                st.download_button(
                    "Download Annotated Video",
                    data=video_bytes,
                    file_name="face_mask_detection.mp4",
                    mime="video/mp4",
                )

            except Exception as error:
                st.error(f"Video processing failed: {error}")

            finally:
                if input_path:
                    Path(input_path).unlink(missing_ok=True)


# --------------------------------------------------
# LIVE WEBCAM
# --------------------------------------------------

class MaskVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = get_model()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        result = self.model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            device="cpu",
            verbose=False,
        )[0]

        annotated = result.plot()

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24",
        )


def webcam_page():
    st.header("Live Webcam Detection")

    st.info(
        "Allow camera access in your browser. "
        "Click START to begin and STOP to finish."
    )

    webrtc_streamer(
        key="face-mask-webcam",
        video_processor_factory=MaskVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )


# --------------------------------------------------
# APPLICATION NAVIGATION
# Keep this section at the bottom
# --------------------------------------------------

image_tab, video_tab, webcam_tab = st.tabs(
    [
        "🖼️ Image Detection",
        "🎥 Video Upload",
        "📷 Live Webcam",
    ]
)

with image_tab:
    image_page()

with video_tab:
    video_page()

with webcam_tab:
    webcam_page()

st.divider()

st.caption(
    "Face Mask Detection | YOLOv8 + MLflow + FastAPI + Streamlit"
)
