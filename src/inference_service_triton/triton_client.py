import time
import logging
import argparse
import numpy as np
import torch
import tifffile as tiff
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient


from itertools import batched
from pathlib import Path
from typing import List

from data_handler.data_bands import get_band_indices
from data_handler.common import preprocess_msi_image

# Default ports for each client type
DEFAULT_PORTS = {
    "http": 8000,
    "grpc": 8001,
}


def get_triton_client(client_type: str, url: str):
    """
    Instantiate and return the appropriate Triton client, InferInput and
    InferRequestedOutput classes for the given client type.

    ---------
    Arguments
    ---------
    client_type: str
        "http" or "grpc"
    url: str
        host:port string for the Triton server

    -------
    Returns
    -------
    (client, InferInput, InferRequestedOutput): tuple
    """
    if client_type == "http":
        triton = httpclient
    elif client_type == "grpc":
        triton = grpcclient
    else:
        raise ValueError(
            f"Unsupported client type '{client_type}'. Choose 'http' or 'grpc'."
        )

    client = triton.InferenceServerClient(url=url)
    return client, triton.InferInput, triton.InferRequestedOutput


def read_tif_as_tensor(
    tif_path: Path,
    list_band_indices: List[int],
) -> np.ndarray:
    """
    Read a multi-band TIF file and return a CHW float32 numpy array
    with the selected bands.

    ---------
    Arguments
    ---------
    tif_path: Path
        full path to the .tif file
    list_band_indices: List[int]
        band indices to select from the HWC raster

    -------
    Returns
    -------
    img_np: np.ndarray
        float32 array of shape (num_bands, H, W)
    """
    img_np = tiff.imread(tif_path)  # HWC
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
    img_np = img_np[list_band_indices, :, :]  # select bands
    return img_np.astype(np.float32)


def run_inference_client(ARGS) -> None:
    logging.basicConfig(
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )

    path_dir_tif_files = Path(ARGS.dir_tif_files)
    list_tif_files_paths = sorted(
        [f for f in path_dir_tif_files.glob(f"*{ARGS.tif_extension}") if f.is_file()]
    )
    num_samples = len(list_tif_files_paths)
    logging.info(f"Found {num_samples} TIF files in {path_dir_tif_files}")

    list_band_indices = get_band_indices(ARGS.data_bands)
    batch_size = ARGS.batch_size

    if batch_size > ARGS.max_batch_size:
        logging.warning(
            f"--batch-size {batch_size} exceeds Triton model max_batch_size "
            f"{ARGS.max_batch_size}. Clamping to {ARGS.max_batch_size}."
        )
        batch_size = ARGS.max_batch_size

    # Connect to Triton Inference Server
    inference_service_url = f"{ARGS.host}:{ARGS.port}"
    client, InferInput, InferRequestedOutput = get_triton_client(
        ARGS.client_type, inference_service_url
    )
    logging.info(
        f"Connected to Triton {ARGS.client_type.upper()} server at {inference_service_url}"
    )

    start_time = time.time()

    for current_batch in batched(list_tif_files_paths, batch_size):
        current_batch = list(current_batch)
        file_names = [f.name for f in current_batch]

        # 1. Read each TIF file in the batch
        batch_arrays = []
        valid_file_names = []
        for tif_path, file_name in zip(current_batch, file_names):
            try:
                img_np = read_tif_as_tensor(tif_path, list_band_indices)
                batch_arrays.append(img_np)
                valid_file_names.append(file_name)
            except Exception as e:
                logging.error(f"Failed to read {file_name}: {e}")

        if not batch_arrays:
            logging.warning("No valid files in batch, skipping.")
            continue

        # 2. Stack into a single batch (N, C, H, W) and normalise
        batch_np = np.stack(batch_arrays, axis=0)  # (N, C, H, W)
        batch_np = preprocess_msi_image(torch.from_numpy(batch_np)).numpy()
        logging.info(f"Sending batch of {len(valid_file_names)} samples to Triton")

        # 3. Build Triton input / output descriptors
        # Input shape: (N, 4, 64, 64) — matches config.pbtxt INPUT__0 dims [4, 64, 64]
        infer_input = InferInput("INPUT__0", batch_np.shape, datatype="FP32")
        infer_input.set_data_from_numpy(batch_np)

        infer_output = InferRequestedOutput("OUTPUT__0")

        # 4. Send request and parse response
        try:
            response = client.infer(
                model_name=ARGS.model_name,
                inputs=[infer_input],
                outputs=[infer_output],
            )

            # Output shape: (N, 10) — raw logits per class
            logits = response.as_numpy("OUTPUT__0")  # (N, 10)
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            confidences = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # softmax
            class_ids = np.argmax(confidences, axis=1)
            top_confidences = confidences[np.arange(len(class_ids)), class_ids]

            for file_name, class_id, confidence in zip(
                valid_file_names, class_ids, top_confidences
            ):
                logging.info(
                    f"File: {file_name} -> Class ID: {class_id} "
                    f"(Confidence: {confidence:.4f})"
                )

        except Exception as e:
            logging.error(
                f"Triton {ARGS.client_type.upper()} inference request failed: {e}"
            )

    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time:.3f} sec.")
    return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triton inference client (HTTP or gRPC) for EuroSAT MSI classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--client-type",
        type=str,
        default="http",
        choices=["http", "grpc"],
        help="Triton client protocol to use. 'http' uses port 8000, 'grpc' uses port 8001",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address of the Triton inference server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port on which the Triton server is listening. "
        "Defaults to 8000 for http and 8001 for grpc if not specified.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="model_reskanet",
        help="Name of the model to query on the Triton server",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of TIF files to send per inference request",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=16,
        help="max_batch_size configured in the Triton model config.pbtxt",
    )
    parser.add_argument(
        "--dir-tif-files",
        type=str,
        required=True,
        help="Full path to the directory containing the TIF files",
    )
    parser.add_argument(
        "--tif-extension",
        type=str,
        default=".tif",
        choices=[".tif", ".tiff"],
        help="File extension to filter TIF files",
    )
    parser.add_argument(
        "--data-bands",
        type=str,
        nargs="*",
        default=["B", "G", "R", "NIR"],
        help="Sentinel-2 bands to select as model input",
    )
    ARGS, _ = parser.parse_known_args()

    # Apply default port based on client type if not explicitly provided
    if ARGS.port is None:
        ARGS.port = DEFAULT_PORTS[ARGS.client_type]

    run_inference_client(ARGS)
    return


if __name__ == "__main__":
    main()
