import torch
import logging
import argparse


from pathlib import Path


from inference.common import load_model_from_checkpoint


def convert_via_jit_trace(ARGS) -> None:
    logging.basicConfig(
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )

    device = torch.device(ARGS.device)
    model_path = Path(ARGS.model_path)
    logging.info(f"Loading model checkpoint from {model_path}")
    loaded_model = load_model_from_checkpoint(
        ARGS.model_path,
        device,
    )
    logging.info("Loaded model checkpoint")

    logging.info("JIT tracing for the model")
    dummy_input = (
        torch.rand(1, ARGS.num_channels, ARGS.img_height, ARGS.img_width)
        .float()
        .to(device)
    )
    traced_model = torch.jit.trace(loaded_model, dummy_input)
    logging.info("JIT tracing for the model completed")

    logging.info("Saving the JIT traced model for Triton serving")
    triton_model_path = Path(ARGS.triton_model_path)
    triton_model_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(triton_model_path)
    logging.info(
        f"Saved the JIT traced model for Triton serving in {triton_model_path}"
    )
    return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert model ckpt to TorchScript format needed for Triton Serving.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num-channels",
        type=int,
        default=4,
        help="Num channels in the input data for the model",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=64,
        help="image height",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=64,
        help="image width",
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
        "--triton-model-path",
        type=str,
        default="./model_for_triton/model.pt",
        help="Absolute or relative path to the ML model file for Triton model serving",
    )

    ARGS, _ = parser.parse_known_args()
    convert_via_jit_trace(ARGS)
    return


if __name__ == "__main__":
    main()
