from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "face_mask_yolov8n_baseline"
    / "weights"
    / "best.pt"
)

DATASET_YAML = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "face_mask_dataset"
    / "data.yaml"
)


def main():

    print("=" * 60)
    print("FACE MASK DETECTION - MODEL EVALUATION")
    print("=" * 60)

    print(f"\nModel:")
    print(MODEL_PATH)

    print(f"\nDataset:")
    print(DATASET_YAML)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"best.pt not found:\n{MODEL_PATH}"
        )

    print("\nLoading trained model...")

    model = YOLO(str(MODEL_PATH))

    print("✅ Model loaded successfully")

    print("\nEvaluating on TEST dataset...")

    results = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=640,
        batch=8,
        device="cpu",
        project=str(PROJECT_ROOT / "runs"),
        name="face_mask_evaluation",
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(f"\nmAP50     : {results.box.map50:.4f}")
    print(f"mAP50-95  : {results.box.map:.4f}")

    print("\nPer-class mAP50:")

    for class_id, score in enumerate(results.box.ap50):

        class_name = model.names[class_id]

        print(
            f"  {class_id} - {class_name}: {score:.4f}"
        )

    print("\nEvaluation results saved to:")

    print(
        PROJECT_ROOT
        / "runs"
        / "face_mask_evaluation"
    )


if __name__ == "__main__":
    main()