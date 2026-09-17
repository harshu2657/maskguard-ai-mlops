from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "Face_Mask_Detection"

REGISTERED_MODEL_NAME = "Face_Mask_Detector"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MLFLOW MODEL REGISTRY")
    print("=" * 60)

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    tracking_uri = f"sqlite:///{MLFLOW_DB}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    print("\nTracking URI:")
    print(tracking_uri)

    # --------------------------------------------------------
    # Find experiment
    # --------------------------------------------------------

    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError(
            f"Experiment not found: {EXPERIMENT_NAME}"
        )

    print("\nExperiment:")
    print(EXPERIMENT_NAME)

    # --------------------------------------------------------
    # Find latest successful run
    # --------------------------------------------------------

    client = MlflowClient()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=10
    )

    if not runs:
        raise RuntimeError(
            "No MLflow runs found."
        )

    selected_run = None

    for run in runs:
        if run.info.status == "FINISHED":
            selected_run = run
            break

    if selected_run is None:
        raise RuntimeError(
            "No completed MLflow run found."
        )

    run_id = selected_run.info.run_id

    print("\nSelected run:")
    print(selected_run.info.run_name)

    print("\nRun ID:")
    print(run_id)

    # --------------------------------------------------------
    # Check artifact
    # --------------------------------------------------------

    artifact_uri = (
        f"runs:/{run_id}/model"
    )

    print("\nModel artifact URI:")
    print(artifact_uri)

    # --------------------------------------------------------
    # Create registered model if needed
    # --------------------------------------------------------

    try:

        registered_model = client.get_registered_model(
            REGISTERED_MODEL_NAME
        )

        print("\n✅ Registered model already exists.")

    except Exception:

        print("\nCreating registered model...")

        registered_model = client.create_registered_model(
            REGISTERED_MODEL_NAME,
            description=(
                "Face Mask Detection YOLOv8 model "
                "managed using MLflow."
            )
        )

        print("\n✅ Registered model created.")

    # --------------------------------------------------------
    # Create model version
    # --------------------------------------------------------

    print("\nCreating model version...")

    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=artifact_uri,
        run_id=run_id,
        description=(
            "YOLOv8n baseline model trained for "
            "face mask detection."
        )
    )

    print("\n" + "=" * 60)
    print("✅ MODEL REGISTERED SUCCESSFULLY")
    print("=" * 60)

    print("\nRegistered Model:")
    print(REGISTERED_MODEL_NAME)

    print("\nVersion:")
    print(model_version.version)

    print("\nRun ID:")
    print(run_id)

    print("\nSource:")
    print(artifact_uri)

    # --------------------------------------------------------
    # Add useful tags
    # --------------------------------------------------------

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        key="model_type",
        value="YOLOv8n"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        key="experiment_type",
        value="baseline"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        key="mAP50",
        value="0.8261"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        key="mAP50_95",
        value="0.5407"
    )

    # --------------------------------------------------------
    # Add alias
    # --------------------------------------------------------

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=model_version.version
    )

    print("\nAlias:")
    print("champion")

    print("\n" + "=" * 60)
    print("REGISTRY SETUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()