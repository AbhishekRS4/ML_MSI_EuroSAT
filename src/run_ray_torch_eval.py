import argparse

from inference.eval_ray_torch import evaluate_model


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained EuroSAT MSI model on validation and test sets using Ray Data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint-path",
        required=True,
        type=str,
        help="full path to the model checkpoint file (.pt or .pth)",
    )
    parser.add_argument(
        "--data-bands",
        nargs="*",
        type=str,
        default=["B", "G", "R", "NIR"],
        choices=["B", "G", "R", "NIR", "SWIR-1", "SWIR-2"],
        help="sentinel-2 bands to use for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="batch size for evaluation",
    )
    parser.add_argument(
        "--dataset-path",
        default="hf://datasets/blanchon/EuroSAT_MSI/data",
        type=str,
        help="HuggingFace dataset path containing parquet files",
    )
    parser.add_argument(
        "--output-log-file",
        default="ray_torch_eval.log",
        type=str,
        help="file path for the evaluation output log",
    )

    ARGS, _ = parser.parse_known_args()
    return ARGS


def main() -> None:
    ARGS = parse_arguments()

    evaluate_model(
        checkpoint_path=ARGS.checkpoint_path,
        data_bands=ARGS.data_bands,
        batch_size=ARGS.batch_size,
        dataset_path=ARGS.dataset_path,
        output_log_file=ARGS.output_log_file,
    )
    return


if __name__ == "__main__":
    main()
