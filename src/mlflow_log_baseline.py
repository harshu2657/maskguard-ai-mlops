from pathlib import Path
import mlflow


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "face_mask_yolov8n_baseline"
    / "weights"
    / "best.pt"
)

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


# ============================================================
# EXPERIMENT DETAILS
# ============================================================

EXPERIMENT_NAME = "Face_Mask_Detection"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MLFLOW - LOG FACE MASK BASELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    print(f"\nModel path:")
    print(MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\n❌ best.pt not found:\n{MODEL_PATH}"
        )

    print("\n✅ best.pt found")

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    tracking_uri = f"sqlite:///{MLFLOW_DB}"

    mlflow.set_tracking_uri(tracking_uri)

    print("\nMLflow tracking URI:")
    print(tracking_uri)

    # --------------------------------------------------------
    # Set experiment
    # --------------------------------------------------------

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\nExperiment:")
    print(EXPERIMENT_NAME)

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="YOLOv8n_Baseline_30_Epochs"
    ):

        # ====================================================
        # PARAMETERS
        # ====================================================

        mlflow.log_param(
            "model",
            "YOLOv8n"
        )

        mlflow.log_param(
            "epochs",
            30
        )

        mlflow.log_param(
            "image_size",
            640
        )

        mlflow.log_param(
            "batch_size",
            8
        )

        mlflow.log_param(
            "device",
            "CPU"
        )

        mlflow.log_param(
            "dataset",
            "Face Mask Dataset"
        )

        mlflow.log_param(
            "classes",
            "bad mask, mask, no mask"
        )

        # ====================================================
        # OVERALL METRICS
        # ====================================================

        mlflow.log_metric(
            "mAP50",
            0.8261
        )

        mlflow.log_metric(
            "mAP50_95",
            0.5407
        )

        mlflow.log_metric(
            "precision",
            0.862
        )

        mlflow.log_metric(
            "recall",
            0.750
        )

        # ====================================================
        # PER-CLASS METRICS
        # ====================================================

        mlflow.log_metric(
            "bad_mask_mAP50",
            0.6115
        )

        mlflow.log_metric(
            "mask_mAP50",
            0.9616
        )

        mlflow.log_metric(
            "no_mask_mAP50",
            0.9053
        )

        mlflow.log_metric(
            "bad_mask_precision",
            0.700
        )

        mlflow.log_metric(
            "bad_mask_recall",
            0.533
        )

        mlflow.log_metric(
            "mask_precision",
            0.956
        )

        mlflow.log_metric(
            "mask_recall",
            0.906
        )

        mlflow.log_metric(
            "no_mask_precision",
            0.929
        )

        mlflow.log_metric(
            "no_mask_recall",
            0.812
        )

        # ====================================================
        # LOG MODEL FILE
        # ====================================================

        print("\nUploading best.pt to MLflow...")

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model"
        )

        # ====================================================
        # TAGS
        # ====================================================

        mlflow.set_tag(
            "project",
            "Face Mask Detection with MLOps"
        )

        mlflow.set_tag(
            "model_type",
            "YOLOv8 Object Detection"
        )

        mlflow.set_tag(
            "experiment_type",
            "Baseline"
        )

        mlflow.set_tag(
            "framework",
            "Ultralytics YOLO"
        )

        print("\n" + "=" * 60)
        print("✅ MLFLOW RUN COMPLETED")
        print("=" * 60)

        print("\nMetrics logged:")
        print("mAP50      : 0.8261")
        print("mAP50-95   : 0.5407")
        print("Precision   : 0.862")
        print("Recall      : 0.750")

        print("\nModel artifact:")
        print("best.pt")

        print("\nOpen MLflow:")
        print("http://127.0.0.1:5001")


if __name__ == "__main__":
    main()