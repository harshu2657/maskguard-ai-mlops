from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_YAML = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "face_mask_dataset"
    / "data.yaml"
)

RUNS_DIR = PROJECT_ROOT / "runs"


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

MODEL_NAME = "yolov8n.pt"

EPOCHS = 30
IMAGE_SIZE = 640
BATCH_SIZE = 8

PROJECT_NAME = str(RUNS_DIR)
EXPERIMENT_NAME = "face_mask_yolov8n_baseline"


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("FACE MASK DETECTION - YOLO TRAINING")
    print("=" * 60)

    print(f"\nDataset YAML:")
    print(DATASET_YAML)

    print(f"\nModel:")
    print(MODEL_NAME)

    print(f"\nEpochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")

    # --------------------------------------------------------
    # Check dataset configuration
    # --------------------------------------------------------

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATASET_YAML}"
        )

    # --------------------------------------------------------
    # Load pretrained YOLO model
    # --------------------------------------------------------

    print("\nLoading pretrained YOLO model...")

    model = YOLO(MODEL_NAME)

    print("✅ Model loaded successfully")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...")
    print("⚠️ Training is running on CPU.")
    print("Please do not close this terminal.")

    results = model.train(
        data=str(DATASET_YAML),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        device="cpu",

        project=PROJECT_NAME,

        name=EXPERIMENT_NAME,

        exist_ok=True,

        pretrained=True,

        patience=10,

        workers=2,

        verbose=True
    )

    # --------------------------------------------------------
    # Training complete
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print("\nTraining results:")
    print(results)

    print("\nBest model should be located at:")

    best_model = (
        RUNS_DIR
        / EXPERIMENT_NAME
        / "weights"
        / "best.pt"
    )

    print(best_model)

    print("\n✅ Next step will be model evaluation + MLflow tracking.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()