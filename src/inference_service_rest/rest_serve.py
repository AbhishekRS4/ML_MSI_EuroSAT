import io
import torch
import logging
import uvicorn
import argparse
import numpy as np
import tifffile as tiff


from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException


from data_handler.data_bands import get_band_indices
from data_handler.common import preprocess_msi_image
from inference.common import load_model_from_checkpoint

# 1. Command Line Arguments
parser = argparse.ArgumentParser(
    description="Start the FastAPI REST ML Inference Server.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host address where the FastAPI inference server will be running",
)
parser.add_argument(
    "--port",
    type=int,
    default=8000,
    help="Port on which the FastAPI inference server will be running",
)
parser.add_argument(
    "--end-point",
    type=str,
    default="predict_batch",
    help="Endpoint on which the FastAPI inference prediction service will be running",
)
parser.add_argument(
    "--model-path",
    type=str,
    required=True,
    help="Absolute or relative path to the serialized ML model file",
)
parser.add_argument(
    "--device",
    action="store_true",
    default="cuda",
    help="Enable CUDA/GPU acceleration if specified",
)
parser.add_argument(
    "--data-bands",
    type=str,
    nargs="*",
    default=["B", "G", "R", "NIR"],
    help="List of data bands to be used",
)
parser.add_argument(
    "--workers",
    type=int,
    default=4,
    help="Number of uvicorn worker processes",
)

ARGS, _ = parser.parse_known_args()

# 2. Initialize FastAPI, Setup logging & Load Model
app = FastAPI(title="Multi-TIFF EuroSAT ML Classification API")

# Check for GPU
device = torch.device(ARGS.device)

# Setup Logging
logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)

logging.info(f"Loading {ARGS.model_path} on {device}...")
# Load a pre-trained model from torchvision (weights argument is standard in newer torch versions)
try:
    model = load_model_from_checkpoint(ARGS.model_path, device)
except AttributeError:
    raise ValueError(f"Model '{ARGS.model_path}' not found")
logging.info("Loaded model from checkpoint")

# Load the metadata needed for model inference serving
list_band_indices = get_band_indices(ARGS.data_bands)


# 3. API Endpoints
@app.post(f"/{ARGS.end_point}")
async def predict_multiple_images(files: List[UploadFile] = File(...)):
    """
    Accepts multiple TIFF image samples, runs batch inference, and returns predictions.
    """
    logging.info(f"FastAPI EuroSAT prediction server running on port {ARGS.port}...")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    tensor_list = []
    file_names = []

    for file in files:
        # Validate file extension
        if not (
            file.filename.lower().endswith(".tif")
            or file.filename.lower().endswith(".tiff")
        ):
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is not a valid TIFF image.",
            )

        try:
            # Read image bytes directly into memory
            contents = await file.read()
            img_np = tiff.imread(io.BytesIO(contents))

            # HWC -> CHW
            # select correct bands as input to the model
            img_np = np.transpose(img_np, axes=(2, 0, 1))[list_band_indices, :, :]

            # Apply transforms
            tensor = torch.from_numpy(img_np).float()
            tensor_list.append(tensor)
            file_names.append(file.filename)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing image {file.filename}: {str(e)}",
            )

    # Stack list into a single batch tensor: [Batch_Size, Channels, Height, Width]
    batch_tensor = torch.stack(tensor_list).to(device)
    batch_tensor = preprocess_msi_image(batch_tensor)
    logging.info(f"Created a batch tensors for {len(tensor_list)} samples")

    # 4. Batch Inference
    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # Get highest class ID and its score for each image
        top_probs, top_classes = torch.topk(probabilities, 1)
    logging.info(f"Computed model predictions for {len(tensor_list)} samples")

    # 5. Format Response
    results = []
    for i in range(len(file_names)):
        results.append(
            {
                "filename": file_names[i],
                "class_id": int(top_classes[i][0]),
                "confidence": float(top_probs[i][0]),
            }
        )
    logging.info(f"Create response message for {len(tensor_list)} samples")
    return {"predictions": results}


if __name__ == "__main__":
    # Start the app using parsed command-line configurations.
    # workers > 1 requires the app to be passed as an import string so uvicorn
    # can re-import it cleanly in each forked worker process.
    uvicorn.run(
        "rest_serve:app",
        host=ARGS.host,
        port=ARGS.port,
        workers=ARGS.workers,
    )
