import argparse

from trainer.train_supervised import train_pipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        help="batch size to use for training",
    )
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        help="num workers to use for data loading",
    )
    parser.add_argument(
        "--val-size",
        default=0.2,
        type=float,
        help="validation size to be used for splitting the dataset",
    )
    parser.add_argument(
        "--num-epochs",
        default=100,
        type=int,
        help="num epochs to train the model",
    )
    parser.add_argument(
        "--dir-dataset",
        default="/home/abhishek/Desktop/datasets/EURO_SAT/EuroSAT_MS",
        type=str,
        help="full directory path to dataset containing MSI rasters in respective class directories",
    )
    parser.add_argument(
        "--exp-name",
        default="eurosat_msi_supervised",
        type=str,
        help="mlflow experiment name",
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
        help="model that needs to be trained",
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
        help="dropout ratio to be used in the dropout layer in the model",
    )
    parser.add_argument(
        "--optimizer-name",
        default="adamw",
        type=str,
        help="optimizer to be used for training",
    )
    parser.add_argument(
        "--data-bands",
        nargs="*",
        type=str,
        default=["B", "G", "R"],
        choices=["B", "G", "R", "NIR", "SWIR-1", "SWIR-2"],
        help="the data bands that needs to used for training the model",
    )
    parser.add_argument(
        "--checkpoint-type",
        type=str,
        default="torch_api",
        choices=["torch_api", "mlflow_api"],
        help="the type of checkpoint that needs to be saved",
    )
    parser.add_argument(
        "--output-log-file",
        type=str,
        default="trainer.log",
        help="full path to the logs file where training logs needs to be recorded",
    )
    parser.add_argument(
        "--model-compile",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="whether to train the model using the compile option to reduce overhead",
    )

    ARGS, unparsed = parser.parse_known_args()
    return ARGS


def main() -> None:
    ARGS = parse_arguments()
    train_pipeline(
        ARGS.dir_dataset,
        ARGS.exp_name,
        ARGS.model_name,
        list_filters=ARGS.list_filters,
        dropout_ratio=ARGS.dropout_ratio,
        optimizer_name=ARGS.optimizer_name,
        num_epochs=ARGS.num_epochs,
        learning_rate=ARGS.learning_rate,
        weight_decay=ARGS.weight_decay,
        batch_size=ARGS.batch_size,
        val_size=ARGS.val_size,
        num_workers=ARGS.num_workers,
        data_bands=ARGS.data_bands,
        checkpoint_type=ARGS.checkpoint_type,
        output_log_file=ARGS.output_log_file,
        model_compile=ARGS.model_compile,
    )
    return


if __name__ == "__main__":
    main()
