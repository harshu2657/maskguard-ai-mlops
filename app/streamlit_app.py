import io
import cv2
import numpy as np
import os
import requests
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MaskGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #F8FAFC;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #64748B;
        margin-bottom: 25px;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        text-align: center;
        min-height: 120px;
    }

    .metric-title {
        font-size: 15px;
        color: #64748B;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
        color: #0F4C81;
        margin-top: 8px;
    }

    .result-card {
        padding: 20px;
        border-radius: 14px;
        margin-top: 15px;
        background-color: white;
        border: 1px solid #E2E8F0;
    }

    .footer {
        text-align: center;
        color: #64748B;
        margin-top: 40px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🛡️ MaskGuard AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Real-Time Face Mask Detection powered by YOLOv8, '
    'FastAPI and MLflow'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.write("**Model:** Face_Mask_Detector")
    st.write("**Version:** champion")
    st.write("**Framework:** YOLOv8")
    st.write("**Backend:** FastAPI")
    st.write("**Tracking:** MLflow")

    st.divider()

    st.subheader("API Status")

    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        if health_response.status_code == 200:

            health_data = health_response.json()

            if health_data.get("model_loaded"):

                st.success("🟢 API Online")
                st.success("🟢 Model Loaded")

            else:

                st.warning("🟡 API Online")
                st.error("🔴 Model Not Loaded")

        else:

            st.error("🔴 API Error")

    except requests.RequestException:

        st.error("🔴 API Offline")

        st.caption(
            "Start FastAPI on port 8000."
        )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📷 Upload an Image")

uploaded_file = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload an image containing one or more people."
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    image_bytes = uploaded_file.getvalue()

    original_image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # --------------------------------------------------------
    # Display original image
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Original Image")

        st.image(
            original_image,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Detect button
    # --------------------------------------------------------

    if st.button(
        "🔍 Detect Face Masks",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Running YOLOv8 detection..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/predict",
                    files={
                        "file": (
                            uploaded_file.name,
                            image_bytes,
                            uploaded_file.type
                        )
                    },
                    timeout=60
                )

                # ------------------------------------------------
                # Handle API errors
                # ------------------------------------------------

                if response.status_code != 200:

                    st.error(
                        f"API Error: "
                        f"{response.status_code}"
                    )

                    st.code(
                        response.text
                    )

                    st.stop()

                data = response.json()

                # ------------------------------------------------
                # OpenCV image for bounding boxes
                # ------------------------------------------------

                image_array = np.array(
                    original_image
                )

                image_cv = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2BGR
                )

                # ------------------------------------------------
                # Detection counters
                # ------------------------------------------------

                mask_count = 0
                no_mask_count = 0
                bad_mask_count = 0

                # ------------------------------------------------
                # Draw detections
                # ------------------------------------------------

                for detection in data.get(
                    "detections",
                    []
                ):

                    class_name = detection[
                        "class_name"
                    ]

                    confidence = detection[
                        "confidence"
                    ]

                    box = detection[
                        "bounding_box"
                    ]

                    x1 = int(box["x1"])
                    y1 = int(box["y1"])
                    x2 = int(box["x2"])
                    y2 = int(box["y2"])

                    # --------------------------------------------
                    # Count classes
                    # --------------------------------------------

                    if class_name == "mask":

                        mask_count += 1

                    elif class_name == "no mask":

                        no_mask_count += 1

                    elif class_name == "bad mask":

                        bad_mask_count += 1

                    # --------------------------------------------
                    # Draw bounding box
                    # --------------------------------------------

                    cv2.rectangle(
                        image_cv,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        3
                    )

                    label = (
                        f"{class_name} "
                        f"{confidence * 100:.1f}%"
                    )

                    # --------------------------------------------
                    # Label background
                    # --------------------------------------------

                    (text_width, text_height), _ = (
                        cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            2
                        )
                    )

                    cv2.rectangle(
                        image_cv,
                        (x1, max(
                            0,
                            y1 - text_height - 12
                        )),
                        (
                            x1 + text_width + 8,
                            y1
                        ),
                        (255, 0, 0),
                        -1
                    )

                    cv2.putText(
                        image_cv,
                        label,
                        (
                            x1 + 4,
                            max(
                                text_height + 2,
                                y1 - 6
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                # ------------------------------------------------
                # Convert back to RGB
                # ------------------------------------------------

                annotated_image = cv2.cvtColor(
                    image_cv,
                    cv2.COLOR_BGR2RGB
                )

                # ------------------------------------------------
                # Display result
                # ------------------------------------------------

                with col2:

                    st.markdown(
                        "### Detection Result"
                    )

                    st.image(
                        annotated_image,
                        use_container_width=True
                    )

                # ------------------------------------------------
                # Metrics
                # ------------------------------------------------

                st.markdown("### 📊 Detection Summary")

                m1, m2, m3, m4 = st.columns(4)

                with m1:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">
                                Total Detections
                            </div>
                            <div class="metric-value">
                                {data.get("detection_count", 0)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with m2:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">
                                😷 Mask
                            </div>
                            <div class="metric-value">
                                {mask_count}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with m3:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">
                                🚫 No Mask
                            </div>
                            <div class="metric-value">
                                {no_mask_count}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with m4:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">
                                ⚠️ Bad Mask
                            </div>
                            <div class="metric-value">
                                {bad_mask_count}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ------------------------------------------------
                # Detection Details
                # ------------------------------------------------

                st.markdown(
                    "### 🔎 Detection Details"
                )

                detections = data.get(
                    "detections",
                    []
                )

                if detections:

                    for index, detection in enumerate(
                        detections,
                        start=1
                    ):

                        class_name = detection[
                            "class_name"
                        ]

                        confidence = detection[
                            "confidence"
                        ]

                        if class_name == "mask":

                            st.success(
                                f"Detection {index}: "
                                f"😷 {class_name.upper()} — "
                                f"{confidence * 100:.2f}%"
                            )

                        elif class_name == "no mask":

                            st.error(
                                f"Detection {index}: "
                                f"🚫 {class_name.upper()} — "
                                f"{confidence * 100:.2f}%"
                            )

                        else:

                            st.warning(
                                f"Detection {index}: "
                                f"⚠️ {class_name.upper()} — "
                                f"{confidence * 100:.2f}%"
                            )

                else:

                    st.info(
                        "No faces detected."
                    )

                # ------------------------------------------------
                # Model information
                # ------------------------------------------------

                st.markdown(
                    "### 🤖 Model Information"
                )

                info1, info2, info3 = st.columns(3)

                with info1:

                    st.info(
                        f"Model\n\n"
                        f"{data.get('model', 'Unknown')}"
                    )

                with info2:

                    st.info(
                        f"Alias\n\n"
                        f"{data.get('model_alias', 'Unknown')}"
                    )

                with info3:

                    st.info(
                        f"Detections\n\n"
                        f"{data.get('detection_count', 0)}"
                    )

            except requests.RequestException as e:

                st.error(
                    "Could not connect to FastAPI."
                )

                st.code(
                    str(e)
                )

            except Exception as e:

                st.error(
                    "Unexpected error occurred."
                )

                st.code(
                    str(e)
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ <b>MaskGuard AI</b><br>
        Face Mask Detection with MLOps<br>
        YOLOv8 • FastAPI • MLflow • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)