import io
import grpc
import torch
import logging
import argparse
import numpy as np
import tifffile as tiff
import concurrent.futures

# Import generated gRPC code
import inference_pb2
import inference_pb2_grpc

from typing import Union, List

from data_handler.data_bands import get_band_indices
from data_handler.common import preprocess_msi_image
from inference.common import load_model_from_checkpoint


class ModelService(inference_pb2_grpc.MSImageClassifierServicer):
    def __init__(
        self,
        model_path: str,
        device: Union[str, None] = None,
        data_bands: List[str] = ["B", "G", "R", "NIR"],
    ):
        self.list_band_indices = get_band_indices(data_bands)

        # Initialize your PyTorch framework
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = load_model_from_checkpoint(model_path, device)
        logging.info("Loaded model from checkpoint")

    def PredictBatch(self, request, context):
        tensors = []
        valid_samples = []

        # 1. Parse and preprocess all samples into an evaluation batch
        for sample in request.samples:
            try:
                # Read TIFF using tifffile to accurately handle geospatial or multi-band TIFFs
                with io.BytesIO(sample.image_bytes) as f:
                    img_np = tiff.imread(f)

                # HWC -> CHW
                # select correct bands as input to the model
                img_np = np.transpose(img_np, axes=(2, 0, 1))[
                    self.list_band_indices, :, :
                ]

                # Apply transforms
                tensor = torch.from_numpy(img_np).float()
                tensors.append(tensor)
                valid_samples.append(sample.sample_id)
            except Exception as e:
                logging.error(f"Error processing sample {sample.sample_id}: {e}")
                # Real-world optimization: handle errors gracefully without crashing the batch

        if not tensors:
            return inference_pb2.BatchInferenceResponse()

        # 2. Collate individual images into a single evaluation batch tensor
        batch_tensor = torch.stack(tensors).to(self.device)
        batch_tensor = preprocess_msi_image(batch_tensor)
        logging.info(f"Created a batch tensors for {len(valid_samples)} samples")

        # 3. PyTorch Inference Execution
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, class_ids = torch.max(probabilities, dim=1)
        logging.info(f"Computed model predictions for {len(valid_samples)} samples")

        # 4. Map outputs back to gRPC response message structures
        response = inference_pb2.BatchInferenceResponse()
        for idx, sample_id in enumerate(valid_samples):
            response.predictions.add(
                sample_id=sample_id,
                class_id=int(class_ids[idx].item()),
                confidence=float(confidences[idx].item()),
            )
        logging.info(f"Create response message for {len(valid_samples)} samples")
        return response


def serve(ARGS: argparse.Namespace) -> None:
    logging.basicConfig(
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )

    # Increase maximum message sizes if transferring massive TIFF payloads
    options = [
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100 MB max request
        ("grpc.max_send_message_length", 50 * 1024 * 1024),
    ]
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=ARGS.workers), options=options
    )
    inference_pb2_grpc.add_MSImageClassifierServicer_to_server(
        ModelService(ARGS.model_path, device=ARGS.device, data_bands=ARGS.data_bands),
        server,
    )
    server_address = f"[::]:{ARGS.port}"
    server.add_insecure_port(server_address)
    logging.info(f"gRPC EuroSAT server running on port {ARGS.port}...")
    server.start()
    server.wait_for_termination()
    return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the gRPC ML Inference Server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port on which the gRPC server will listen",
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
        "--workers",
        type=int,
        default=4,
        help="Number of thread workers for the gRPC pool",
    )
    parser.add_argument(
        "--data-bands",
        type=str,
        nargs="*",
        default=["B", "G", "R", "NIR"],
        help="List of data bands to be used",
    )
    ARGS, _ = parser.parse_known_args()

    serve(ARGS)
    return


if __name__ == "__main__":
    main()
