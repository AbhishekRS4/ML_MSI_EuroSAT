import logging
import argparse

from trainer.train_ray_torch import train_pipeline_ray


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EuroSAT MSI model using Ray TorchTrainer with streaming HuggingFace dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--learning-rate",
        default=1e-3,
        type=float,
        help="learning rate to use for training",
    )
    parser.add_argument(
        "--weight-decay",
        default=5e-5,
        type=float,
        help="weight decay to use for training",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="batch size per worker to use for training",
    )
    parser.add_argument(
        "--num-epochs",
        default=100,
        type=int,
        help="number of epochs to train the model",
    )
    parser.add_argument(
        "--dataset-path",
        default="hf://datasets/blanchon/EuroSAT_MSI/data",
        type=str,
        help="HuggingFace dataset path containing parquet files",
    )
    parser.add_argument(
        "--model-name",
        default="resnet",
        type=str,
        choices=[
            "resnet",
            "reskanet",
            "se_resnet",
            "se_reskanet",
            "psa_resnet",
            "psa_reskanet",
        ],
        help="model architecture to train",
    )
    parser.add_argument(
        "--list-filters",
        nargs="*",
        type=int,
        default=[64, 128, 256],
        help="list of num filters to be used in the model",
    )
    parser.add_argument(
        "--dropout-ratio",
        type=float,
        default=0.2,
        help="dropout ratio to be used in the dropout layer",
    )
    parser.add_argument(
        "--loss-fn",
        default="cross_entropy",
        type=str,
        choices=["cross_entropy", "focal"],
        help="loss function to be used for training",
    )
    parser.add_argument(
        "--data-bands",
        nargs="*",
        type=str,
        default=["B", "G", "R", "NIR"],
        choices=["B", "G", "R", "NIR", "SWIR-1", "SWIR-2"],
        help="the data bands to use for training the model",
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="number of Ray Train workers (for distributed training)",
    )
    parser.add_argument(
        "--use-gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="whether to use GPU for training",
    )
    parser.add_argument(
        "--num-cpu-workers",
        default=4,
        type=int,
        help="number of Ray CPU workers (for distributed training)",
    )
    parser.add_argument(
        "--model-compile",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="whether to compile the model using torch.compile",
    )
    parser.add_argument(
        "--run-name",
        default="eurosat_msi_ray_train",
        type=str,
        help="name for the Ray Train run",
    )
    parser.add_argument(
        "--exp-name",
        default="eurosat_msi_ray_train",
        type=str,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        type=str,
        help="MLflow tracking URI (default: None, uses local file store)",
    )
    parser.add_argument(
        "--output-log-file",
        default="ray_torch_trainer.log",
        type=str,
        help="full path to the log file where training logs are recorded",
    )

    ARGS, _ = parser.parse_known_args()
    return ARGS


def setup_logging(output_log_file: str) -> None:
    """
    Configure logging to write to both a file and the console.

    ---------
    Arguments
    ---------
    output_log_file: str
        path to the output log file
    """
    logging.basicConfig(
        filename=output_log_file,
        filemode="a",
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logging.getLogger().addHandler(console_handler)


def main() -> None:
    ARGS = parse_arguments()

    setup_logging(ARGS.output_log_file)

    train_pipeline_ray(
        model_name=ARGS.model_name,
        list_filters=ARGS.list_filters,
        dropout_ratio=ARGS.dropout_ratio,
        loss_fn=ARGS.loss_fn,
        num_epochs=ARGS.num_epochs,
        learning_rate=ARGS.learning_rate,
        weight_decay=ARGS.weight_decay,
        batch_size=ARGS.batch_size,
        data_bands=ARGS.data_bands,
        num_workers=ARGS.num_workers,
        use_gpu=ARGS.use_gpu,
        num_cpu_workers=ARGS.num_cpu_workers,
        model_compile=ARGS.model_compile,
        dataset_path=ARGS.dataset_path,
        exp_name=ARGS.exp_name,
        mlflow_tracking_uri=ARGS.mlflow_tracking_uri,
        run_name=ARGS.run_name,
    )
    return


if __name__ == "__main__":
    main()
