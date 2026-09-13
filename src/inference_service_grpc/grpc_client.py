import os
import grpc
import time
import logging
import argparse


from itertools import batched
from pathlib import Path, PosixPath


from inference_service_grpc import inference_pb2
from inference_service_grpc import inference_pb2_grpc


def run_inference_client(ARGS):
    logging.basicConfig(
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )

    # Max payload configs matching the server specification
    options = [
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
    ]

    server_address = f"localhost:{ARGS.port}"

    path_dir_tif_files = Path(ARGS.dir_tif_files)

    list_tif_files_paths = sorted(
        [f for f in path_dir_tif_files.glob(f"*{ARGS.tif_extension}") if f.is_file()]
    )
    num_samples = len(list_tif_files_paths)
    batch_size = ARGS.batch_size

    # Establish connection channel
    start_time = time.time()
    with grpc.insecure_channel(server_address, options=options) as channel:
        stub = inference_pb2_grpc.MSImageClassifierStub(channel)

        for current_batch in batched(list_tif_files_paths, batch_size):
            # Build batch request array
            request = inference_pb2.BatchInferenceRequest()

            for tif_file_path in current_batch:
                if os.path.exists(tif_file_path):
                    with open(tif_file_path, "rb") as f:
                        file_bytes = f.read()

                    # Append sample to batch
                    sample = inference_pb2.MSImageSample(
                        image_bytes=file_bytes,
                        sample_id=os.path.basename(tif_file_path),
                    )
                    request.samples.append(sample)

            if not request.samples:
                logging.info("No valid files supplied.")
                return

            logging.info(
                f"Sending batch request with {len(request.samples)} .tif files..."
            )

            # Fire gRPC remote call
            try:
                response = stub.PredictBatch(request)

                # Print batch output predictions
                logging.info("\nInference Results:")
                for pred in response.predictions:
                    logging.info(
                        f"File: {pred.sample_id} -> Class ID: {pred.class_id} (Confidence: {pred.confidence:.4f})"
                    )

            except grpc.RpcError as e:
                logging.info(f"gRPC call failed: {e.code()} - {e.details()}")

    end_time = time.time()
    time_taken = end_time - start_time
    logging.info(f"Total time taken: {time_taken} sec.")
    return


def main() -> None:
    # Mocking execution using dummy paths
    # Replace these paths with actual local .tif file paths for testing
    parser = argparse.ArgumentParser(
        description="Start the gRPC ML Inference Client.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port on which the gRPC server will listen",
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
