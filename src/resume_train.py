from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LAST_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "face_mask_yolov8n_baseline"
    / "weights"
    / "last.pt"
)

print("Resuming from:")
print(LAST_MODEL)

model = YOLO(str(LAST_MODEL))

model.train(resume=True)