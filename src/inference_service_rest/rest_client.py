import os
import time
import logging
import argparse
import requests


from itertools import batched
from pathlib import Path, PosixPath


def run_inference_client(ARGS):
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
    batch_size = ARGS.batch_size

    inference_service_url = f"{ARGS.host}:{ARGS.port}/{ARGS.end_point}/"

    start_time = time.time()
    for current_batch in batched(list_tif_files_paths, batch_size):
        files = [
            ("files", (f.name, open(f, "rb"), "image/tiff")) for f in current_batch
        ]
        try:
            response = requests.post(inference_service_url, files=files)
            for pred in response.json()["predictions"]:
                logging.info(
                    f"File: {pred['filename']} -> Class ID: {pred['class_id']} (Confidence: {pred['confidence']:.4f})"
                )
        finally:
            for _, (_, fd, _) in files:
                fd.close()

    end_time = time.time()
    time_taken = end_time - start_time
    logging.info(f"Total time taken: {time_taken} sec.")
    return


def main() -> None:
    # Mocking execution using dummy paths
    # Replace these paths with actual local .tif file paths for testing
    parser = argparse.ArgumentParser(
        description="Start the FastAPI REST ML Inference Client.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port on which the FastAPI inference server will listen",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://0.0.0.0",
        help="Host address where the FastAPI inference serve will be running",
    )
    parser.add_argument(
        "--end-point",
        type=str,
        default="predict_batch",
        help="Endpoint on which the FastAPI inference prediction service will be running",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="batch size to be used for inference, accordingly messages will be sent",
    )
    parser.add_argument(
        "--dir-tif-files",
        type=str,
        default="./dir_with_tif_files/",
        help="Full path to the directory where the test TIF files are present",
    )
    parser.add_argument(
        "--tif-extension",
        type=str,
        default=".tif",
        choices=[".tif", ".tiff"],
        help="Port on which the gRPC server will listen",
    )
    ARGS, _ = parser.parse_known_args()

    run_inference_client(ARGS)
    return


if __name__ == "__main__":
    main()
