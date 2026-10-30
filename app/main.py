from pathlib import Path
import os
import tempfile

import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLFLOW_DB = PROJECT_ROOT / "mlflow_test_restore.db"

MODEL_NAME = "Face_Mask_Detector"
MODEL_ALIAS = "champion"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Face Mask Detection API",
    description="Real-time face mask detection API using YOLOv8 and MLflow.",
    version="1.0.0",
)

model = None


# ============================================================
# GET MODEL FROM MLFLOW
# ============================================================

def get_model_path_from_mlflow():

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{MLFLOW_DB}"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    print("\nMLflow URI:")
    print(tracking_uri)

    print("\nModel URI:")
    print(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

    client = MlflowClient()

    # --------------------------------------------------------
    # Find model version using champion alias
    # --------------------------------------------------------

    model_version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS
    )

    print("\nResolved model:")
    print(f"  Name: {model_version.name}")
    print(f"  Version: {model_version.version}")
    print(f"  Run ID: {model_version.run_id}")
    print(f"  Source: {model_version.source}")

    run_id = model_version.run_id

    # --------------------------------------------------------
    # Get artifact information
    # --------------------------------------------------------

    run = client.get_run(run_id)

    print("\nActual artifact URI:")
    print(run.info.artifact_uri)

    # --------------------------------------------------------
    # Ask MLflow for the actual artifact file
    # instead of manually constructing a Windows file path
    # --------------------------------------------------------

    print("\nLooking for model artifact...")

    try:

        model_path = client.download_artifacts(
            run_id,
            "model/best.pt"
        )

    except Exception as e:

        print("\nCould not download/find model artifact.")

        print("\nTrying to list available artifacts...")

        try:

            artifacts = client.list_artifacts(run_id)

            for artifact in artifacts:

                print(f"  {artifact.path}")

                if artifact.is_dir:

                    try:

                        sub_artifacts = client.list_artifacts(
                            run_id,
                            artifact.path
                        )

                        for sub_artifact in sub_artifacts:
                            print(
                                f"    {sub_artifact.path}"
                            )

                    except Exception:
                        pass

        except Exception as list_error:

            print(
                f"Could not list artifacts: {list_error}"
            )

        raise FileNotFoundError(
            "Could not find model/best.pt in MLflow.\n"
            f"Original error: {e}"
        )

    model_path = Path(model_path)

    print("\nDownloaded/local model path:")
    print(model_path)

    if not model_path.exists():

        raise FileNotFoundError(
            "MLflow returned a model path, "
            "but the file does not exist:\n"
            f"{model_path}"
        )

    print("\nMODEL FILE FOUND")

    return model_path


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global model

    print("=" * 60)
    print("LOADING FACE MASK MODEL")
    print("=" * 60)

    try:

        model_path = get_model_path_from_mlflow()

        print("\nLoading YOLO model...")
        print(f"Path: {model_path}")

        model = YOLO(str(model_path))

        print("\nYOLO model loaded successfully")

        print("\nClasses:")

        for class_id, class_name in model.names.items():

            print(
                f"  {class_id}: {class_name}"
            )

        print("\n" + "=" * 60)
        print("MODEL LOADING COMPLETE")
        print("=" * 60)

    except Exception as e:

        print("\nMODEL LOADING FAILED")
        print(str(e))

        raise


# ============================================================
# STARTUP EVENT
# ============================================================

@app.on_event("startup")
def startup_event():

    load_model()


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Face Mask Detection API",
        "status": "running",
        "model": MODEL_NAME,
        "version": MODEL_ALIAS,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    if model is None:

        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False
            }
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp",
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Use JPG, PNG, or WEBP."
        )

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    suffix = Path(
        file.filename or ".jpg"
    ).suffix

    temp_path = None

    try:

        # ----------------------------------------------------
        # Create temporary image
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(image_bytes)

            temp_path = temp_file.name

        # ----------------------------------------------------
        # YOLO prediction
        # ----------------------------------------------------

        results = model.predict(
            source=temp_path,
            imgsz=640,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        result = results[0]

        detections = []

        # ----------------------------------------------------
        # Process detections
        # ----------------------------------------------------

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )

                detections.append(
                    {
                        "class_id": class_id,

                        "class_name": model.names[
                            class_id
                        ],

                        "confidence": round(
                            confidence,
                            4
                        ),

                        "bounding_box": {

                            "x1": round(
                                coordinates[0],
                                2
                            ),

                            "y1": round(
                                coordinates[1],
                                2
                            ),

                            "x2": round(
                                coordinates[2],
                                2
                            ),

                            "y2": round(
                                coordinates[3],
                                2
                            ),
                        }
                    }
                )

        # ----------------------------------------------------
        # Return prediction
        # ----------------------------------------------------

        return {

            "success": True,

            "model": MODEL_NAME,

            "model_alias": MODEL_ALIAS,

            "filename": file.filename,

            "detection_count": len(
                detections
            ),

            "detections": detections,
        }

    finally:

        # ----------------------------------------------------
        # Delete temporary image
        # ----------------------------------------------------

        if temp_path:

            temp_file_path = Path(
                temp_path
            )

            if temp_file_path.exists():

                temp_file_path.unlink()


# ============================================================
# RUN APPLICATION DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )