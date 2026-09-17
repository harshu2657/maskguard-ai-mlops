from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "Face_Mask_Detection"

def main():

    print("=" * 60)
    print("MLFLOW SETUP")
    print("=" * 60)

    tracking_uri = f"sqlite:///{MLFLOW_DB}"

    print(f"\nMLflow database:")
    print(MLFLOW_DB)

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print("\n✅ New MLflow experiment created")
    else:
        experiment_id = experiment.experiment_id
        print("\n✅ Existing MLflow experiment found")

    print(f"Experiment name : {EXPERIMENT_NAME}")
    print(f"Experiment ID   : {experiment_id}")

    client = MlflowClient()

    print("\nMLflow tracking URI:")
    print(tracking_uri)

    print("\n" + "=" * 60)
    print("MLFLOW SETUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()