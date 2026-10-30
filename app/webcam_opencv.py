
import cv2

from app.video_utils import load_detection_model


def main():
    print("Loading Face Mask Detection model...")
    model = load_detection_model()
    print("Model loaded successfully!")
    print("Press Q or ESC to stop webcam detection.")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Cannot open webcam. Check camera permissions "
            "and close other applications using the camera."
        )

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a webcam frame.")
                break

            result = model.predict(
                source=frame,
                imgsz=640,
                conf=0.25,
                device="cpu",
                verbose=False,
            )[0]

            annotated = result.plot()

            cv2.imshow(
                "Face Mask Detection - Press Q to quit",
                annotated,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
