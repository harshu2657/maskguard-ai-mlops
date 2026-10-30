
import cv2
import tempfile
from pathlib import Path

from ultralytics import YOLO

from main import get_model_path_from_mlflow


def load_detection_model():
    """Load the existing trained model through MLflow."""
    model_path = get_model_path_from_mlflow()
    return YOLO(str(model_path))


def process_video(input_path, model, confidence=0.25):
    """
    Process an uploaded video and save an annotated MP4.
    Returns output_path and aggregate detection counts.
    """
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise ValueError(
            "Cannot open video. Try a valid MP4 or AVI file."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4"
    )
    output_path = output_file.name
    output_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path, fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        cap.release()
        Path(output_path).unlink(missing_ok=True)
        raise ValueError("Could not create output video.")

    counts = {name: 0 for name in model.names.values()}
    frames_processed = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            result = model.predict(
                source=frame,
                imgsz=640,
                conf=confidence,
                device="cpu",
                verbose=False,
            )[0]

            if result.boxes is not None:
                for class_id in result.boxes.cls.tolist():
                    name = model.names[int(class_id)]
                    counts[name] = counts.get(name, 0) + 1

            annotated_frame = result.plot()
            writer.write(annotated_frame)
            frames_processed += 1

    except Exception:
        Path(output_path).unlink(missing_ok=True)
        raise

    finally:
        cap.release()
        writer.release()

    if frames_processed == 0:
        Path(output_path).unlink(missing_ok=True)
        raise ValueError("No readable frames were found in the video.")

    return output_path, counts, frames_processed, total_frames
