from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

MODEL_NAME = "Face_Mask_Detector"


def main():

    print("=" * 60)
    print("MLFLOW REGISTRY CLEANUP")
    print("=" * 60)

    tracking_uri = f"sqlite:///{MLFLOW_DB}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = MlflowClient()

    # --------------------------------------------------------
    # Delete duplicate Version 2
    # --------------------------------------------------------

    print("\nDeleting duplicate Version 2...")

    try:
        client.delete_model_version(
            name=MODEL_NAME,
            version="2"
        )

        print("✅ Version 2 deleted")

    except Exception as e:
        print(f"⚠️ Could not delete Version 2: {e}")

    # --------------------------------------------------------
    # Set Version 1 as champion
    # --------------------------------------------------------

    print("\nSetting Version 1 as champion...")

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version="1"
    )

    print("✅ Version 1 is now champion")

    # --------------------------------------------------------
    # Display final registry state
    # --------------------------------------------------------

    model = client.get_registered_model(MODEL_NAME)

    print("\n" + "=" * 60)
    print("REGISTRY CLEANUP COMPLETE")
    print("=" * 60)

    print(f"\nModel: {model.name}")

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    print("\nModel versions:")

    for version in versions:
        print(
            f"  Version {version.version} "
            f"| Run: {version.run_id}"
        )

    print("\nChampion:")
    print("  Version 1")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()