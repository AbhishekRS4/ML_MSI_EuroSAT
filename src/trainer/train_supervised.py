import sys
import time
import torch
import shutil
import mlflow
import logging
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
from copy import deepcopy
from torch import GradScaler
from torch.optim import AdamW
from typing import Union, List, Tuple
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import PolynomialLR
from sklearn.metrics import ConfusionMatrixDisplay
from mlflow.models.signature import infer_signature

from data_handler.file_utils import get_list_files_n_labels
from data_handler.supervised_data_loader import get_dataloaders_for_training
from metrics.compute_metrics import MetricsCalculator, get_confusion_matrix_figure
from models.msi_supervised import (
    MSI_ResNet,
    MSI_ResKANet,
    MSI_SE_ResNet,
    MSI_SE_ResKANet,
    MSI_PSA_ResNet,
    MSI_PSA_ResKANet,
)


def train_model(
    model: Union[
        MSI_ResNet,
        MSI_ResKANet,
        MSI_SE_ResNet,
        MSI_SE_ResKANet,
        MSI_PSA_ResNet,
        MSI_PSA_ResKANet,
    ],
    optimizer: AdamW,
    criterion: CrossEntropyLoss,
    train_loader: DataLoader,
    device: torch.device,
    scaler: GradScaler,
    metrics_calculator: MetricsCalculator,
) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    """
    train loop

    ---------
    Arguments
    ---------
    model: Union[MSI_ResNet, MSI_ResKANet, MSI_SE_ResNet, MSI_SE_ResKANet, MSI_PSA_ResNet, MSI_PSA_ResKANet]
        a valid object of type torch model
    optimizer: AdamW
        a valid object of type torch Optimizer
    criterion: CrossEntropyLoss
        a valid object of type torch criterion function
    train_loader: DataLoader
        an object of type torch dataloader
    device: torch.device
        a valid object of type torch device
    scaler: GradScaler
        a valid object of type GradScaler
    metrics_calculator: MetricsCalculator
        a valid object of type MetricsCalculator

    -------
    Returns
    -------
    train_loss, train_acc, train_f1, train_prec, train_rec, train_conf_mat_row_norm, train_conf_mat_col_norm:
    Tuple[float, float, float, float, float, np.ndarray, np.ndarray]
        a tuple of training loss, accuracy, f1-score, precision, recall and confusion matrices
    """
    model.to(device)
    model.train()
    num_train_batches = len(train_loader)

    running_train_loss = torch.zeros(1).to(device)
    running_train_acc = torch.zeros(1).to(device)
    running_train_f1 = torch.zeros(1).to(device)
    running_train_prec = torch.zeros(1).to(device)
    running_train_rec = torch.zeros(1).to(device)
    metrics_calculator.reset_confusion_matrix()

    for msi_bands, gt_labels in train_loader:
        msi_bands = msi_bands.to(device, dtype=torch.float)
        gt_labels = gt_labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        # run forward pass with autocast
        with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
            pred_logits = model(msi_bands)
            loss = criterion(pred_logits, gt_labels)
            pred_labels = torch.argmax(pred_logits, dim=1)

            acc, f1, prec, rec = metrics_calculator.compute_base_metrics(
                gt_labels, pred_labels
            )
            running_train_loss += loss
            running_train_acc += acc
            running_train_f1 += f1
            running_train_prec += prec
            running_train_rec += rec
            metrics_calculator.update_confusion_matrix(gt_labels, pred_labels)

        # loss is scaled and then scaled gradients are created
        scaler.scale(loss).backward()
        # apply the update
        scaler.step(optimizer)
        # update the scaler for the next iteration
        scaler.update()

    train_loss = float(running_train_loss) / num_train_batches
    train_acc = float(running_train_acc) / num_train_batches
    train_f1 = float(running_train_f1) / num_train_batches
    train_prec = float(running_train_prec) / num_train_batches
    train_rec = float(running_train_rec) / num_train_batches

    train_conf_mat_row_norm, train_conf_mat_col_norm = (
        metrics_calculator.compute_confusion_matrix()
    )
    train_conf_mat_row_norm = train_conf_mat_row_norm.clone().detach().cpu().numpy()
    train_conf_mat_col_norm = train_conf_mat_col_norm.clone().detach().cpu().numpy()

    return (
        train_loss,
        train_acc,
        train_f1,
        train_prec,
        train_rec,
        train_conf_mat_row_norm,
        train_conf_mat_col_norm,
    )


def test_model(
    model: Union[
        MSI_ResNet,
        MSI_ResKANet,
        MSI_SE_ResNet,
        MSI_SE_ResKANet,
        MSI_PSA_ResNet,
        MSI_PSA_ResKANet,
    ],
    criterion: CrossEntropyLoss,
    test_loader: DataLoader,
    device: torch.device,
    metrics_calculator: MetricsCalculator,
) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    """
    test loop

    ---------
    Arguments
    ---------
    model: Union[MSI_ResNet, MSI_ResKANet,]
        an object of type torch model
    criterion: CrossEntropyLoss
        an object of type torch criterion function
    test_loader: DataLoader
        an object of type torch dataloader
    device: torch.device
        an object of type torch device
    metrics_calculator: MetricsCalculator
        a valid object of type MetricsCalculator

    -------
    Returns
    -------
    (test_loss, test_acc, test_f1, test_prec, test_rec, test_conf_mat_row_norm, test_conf_mat_col_norm):
    Tuple[float, float, float, float, float, np.ndarray, np.ndarray]
        a tuple of training loss, accuracy, f1-score, precision, recall and confusion matrices
    """
    model.to(device)
    model.eval()
    num_test_batches = len(test_loader)

    running_test_loss = torch.zeros(1).to(device)
    running_test_acc = torch.zeros(1).to(device)
    running_test_f1 = torch.zeros(1).to(device)
    running_test_prec = torch.zeros(1).to(device)
    running_test_rec = torch.zeros(1).to(device)
    metrics_calculator.reset_confusion_matrix()

    with torch.no_grad():
        for msi_bands, gt_labels in test_loader:
            msi_bands = msi_bands.to(device, dtype=torch.float)
            gt_labels = gt_labels.to(device, dtype=torch.long)

            pred_logits = model(msi_bands)
            loss = criterion(pred_logits, gt_labels)
            pred_labels = torch.argmax(pred_logits, dim=1)

            acc, f1, prec, rec = metrics_calculator.compute_base_metrics(
                gt_labels, pred_labels
            )
            running_test_loss += loss
            running_test_acc += acc
            running_test_f1 += f1
            running_test_prec += prec
            running_test_rec += rec
            metrics_calculator.update_confusion_matrix(gt_labels, pred_labels)

    test_loss = float(running_test_loss) / num_test_batches
    test_acc = float(running_test_acc) / num_test_batches
    test_f1 = float(running_test_f1) / num_test_batches
    test_prec = float(running_test_prec) / num_test_batches
    test_rec = float(running_test_rec) / num_test_batches

    test_conf_mat_row_norm, test_conf_mat_col_norm = (
        metrics_calculator.compute_confusion_matrix()
    )
    test_conf_mat_row_norm = test_conf_mat_row_norm.clone().detach().cpu().numpy()
    test_conf_mat_col_norm = test_conf_mat_col_norm.clone().detach().cpu().numpy()

    return (
        test_loss,
        test_acc,
        test_f1,
        test_prec,
        test_rec,
        test_conf_mat_row_norm,
        test_conf_mat_col_norm,
    )


def train_pipeline(
    dir_dataset: str,
    exp_name: str,
    model_name: str,
    list_filters: List[int] = [64, 128, 256],
    dropout_ratio: float = 0.2,
    optimizer_name: str = "adamw",
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-5,
    batch_size: int = 64,
    val_size: float = 0.2,
    num_workers: int = 8,
    data_bands: List[str] = ["B", "G", "R"],
    checkpoint_type: str = "torch_api",
    output_log_file: str = "trainer.log",
    model_compile: bool = True,
) -> None:
    """
    main training pipeline for supervised training

    ---------
    Arguments
    ---------

    dir_dataset: str
        full path to directory containing the dataset
    exp_name: str
        experiment name to be used in MLFlow
    model_name: str
        model name
    list_filters: List[int]
        list of filters to be used in the model(default: [64, 128, 256])
    dropout_ratio: float
        dropout ratio to be used in the dropout layer (default: 0.2)
    optimizer_name: str
        optimizer to be used for optimization during training (default: adamw)
    num_epochs: int
        number of epochs for which the model needs to be trained (default: 100)
    learning_rate: float
        learning rate to be used for training (default: 1e-3)
    weight_decay: float
        weight decay to be used for training (default: 5e-5)
    batch_size: int
        batch size to be used for training (default: 64)
    val_size: float
        validation set size (default: 0.2)
    num_workers: int
        number of workers to be used for data loading (default: 8)
    data_bands: List[str]
        a list of data bands that needs to be used for training the model (default: [B, G, R])
    torch_api: str
        indicating the type of the API used to save checkpoints (default: torch_api)
    output_log_file: str
        file name for the output log file (default: trainer.log)
    model_compile: bool
        whether to use the option of compiling the model to reduce overhead during training the model
    """
    if checkpoint_type == "torch_api":
        dir_ckpt_models = "tmp_models"
        path_ckpt_models = Path(dir_ckpt_models)

        if not path_ckpt_models.is_dir():
            path_ckpt_models.mkdir()

    logging.basicConfig(
        filename=output_log_file,
        filemode="a",
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        logging.info("CUDA device not found, so exiting....")
        sys.exit(0)

    logging.info(
        "Training a supervised classification CNN model for the EuroSAT MSI dataset"
    )

    num_input_bands = len(data_bands)
    list_imgs, list_lbls, list_class_names = get_list_files_n_labels(dir_dataset)
    num_classes = list_lbls[-1] + 1

    num_data_samples = len(list_imgs)
    logging.info(f"Num data samples: {num_data_samples}")

    train_loader, val_loader, list_train_imgs, list_val_imgs = (
        get_dataloaders_for_training(
            list_imgs,
            list_lbls,
            data_bands,
            val_size=val_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )

    if model_name == "resnet":
        model = MSI_ResNet(
            num_input_bands=num_input_bands,
            num_classes=num_classes,
            list_filters=list_filters,
            dropout_ratio=dropout_ratio,
        )
    elif model_name == "reskanet":
        model = MSI_ResKANet(
            num_input_bands=num_input_bands,
            num_classes=num_classes,
            list_filters=list_filters,
            dropout_ratio=dropout_ratio,
        )
    elif model_name == "se_resnet":
        model = MSI_SE_ResNet(
            num_input_bands=num_input_bands,
            num_classes=num_classes,
            list_filters=list_filters,
            dropout_ratio=dropout_ratio,
        )
    elif model_name == "se_reskanet":
        model = MSI_SE_ResKANet(
            num_input_bands=num_input_bands,
            num_classes=num_classes,
            list_filters=list_filters,
            dropout_ratio=dropout_ratio,
        )
    elif model_name == "psa_resnet":
        model = MSI_PSA_ResNet(
            num_input_bands=num_input_bands,
            num_classes=num_classes,
            list_filters=list_filters,
            dropout_ratio=dropout_ratio,
        )
    elif model_name == "psa_reskanet":
        model = MSI_PSA_ResKANet(
            num_input_bands=num_input_bands,
            num_classes=num_classes,
            list_filters=list_filters,
            dropout_ratio=dropout_ratio,
        )
    else:
        logging.info(f"Unidentified option for arg (model_name): {model_name}")
    model.to(device)
    if model_compile:
        model = torch.compile(model, mode="reduce-overhead")

    metrics_calculator = MetricsCalculator(device, num_classes=num_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    lr_scheduler = PolynomialLR(
        optimizer,
        num_epochs + 1,
        power=0.95,
    )

    criterion = CrossEntropyLoss()
    scaler = GradScaler()

    logging.info(
        f"Training the EuroSAT MSI classification model started, model_name: {model_name}, num_classes: {num_classes}"
    )

    best_val_acc = 0

    mlflow_run_tags = {
        "experiment": exp_name,
        "run_name": model_name,
    }

    mlflow.set_experiment(exp_name)
    experiment = mlflow.get_experiment_by_name(exp_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id, tags=mlflow_run_tags):
        mlflow.log_param("optimization.optimizer_name", optimizer_name)
        mlflow.log_param("optimization.num_epochs", num_epochs)
        mlflow.log_param("optimization.learning_rate", learning_rate)
        mlflow.log_param("optimization.weight_decay", weight_decay)
        mlflow.log_param("optimization.batch_size", batch_size)

        mlflow.log_param("dataset.data_bands", data_bands)
        mlflow.log_param("dataset.num_classes", num_classes)
        mlflow.log_param("dataset.dir_dataset", dir_dataset)
        mlflow.log_param("dataset.list_class_names", list_class_names)
        mlflow.log_param("dataset.validation_size", val_size)
        mlflow.log_text("\n".join(list_train_imgs), "dataset_list_train_images.txt")
        mlflow.log_text("\n".join(list_val_imgs), "dataset_list_val_images.txt")

        mlflow.log_param("model.model_name", model_name)
        mlflow.log_param("model.list_filters", list_filters)
        mlflow.log_param("model.num_input_bands", num_input_bands)
        mlflow.log_param("model.dropout_ratio", dropout_ratio)

        for epoch in range(1, num_epochs + 1):
            time_start = time.time()
            (
                train_loss,
                train_acc,
                train_f1,
                train_precision,
                train_recall,
                train_conf_mat_row_norm,
                train_conf_mat_col_norm,
            ) = train_model(
                model,
                optimizer,
                criterion,
                train_loader,
                device,
                scaler,
                metrics_calculator,
            )
            (
                val_loss,
                val_acc,
                val_f1,
                val_precision,
                val_recall,
                val_conf_mat_row_norm,
                val_conf_mat_col_norm,
            ) = test_model(
                model,
                criterion,
                val_loader,
                device,
                metrics_calculator,
            )
            lr_scheduler.step()
            time_end = time.time()
            logging.info(
                f"Epoch: {epoch}/{num_epochs}, time: {time_end-time_start:.4f} sec."
            )
            logging.info(
                f"Train set, loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, f1: {train_f1:.4f}, precision: {train_precision:.4f}, recall: {train_recall:.4f}"
            )
            logging.info(
                f"Validation set, loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, f1: {val_f1:.4f}, precision: {val_precision:.4f}, recall: {val_recall:.4f}\n"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("train_precision", train_precision, step=epoch)
            mlflow.log_metric("train_recall", train_recall, step=epoch)

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)
            mlflow.log_metric("val_precision", val_precision, step=epoch)
            mlflow.log_metric("val_recall", val_recall, step=epoch)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                # get the confusion matrix figures
                train_conf_mat_row_norm_fig = get_confusion_matrix_figure(
                    train_conf_mat_row_norm, list_class_names
                )
                train_conf_mat_col_norm_fig = get_confusion_matrix_figure(
                    train_conf_mat_col_norm, list_class_names
                )
                val_conf_mat_row_norm_fig = get_confusion_matrix_figure(
                    val_conf_mat_row_norm, list_class_names
                )
                val_conf_mat_col_norm_fig = get_confusion_matrix_figure(
                    val_conf_mat_col_norm, list_class_names
                )

                # log all the confusion matrix figures
                mlflow.log_figure(
                    train_conf_mat_row_norm_fig.figure_,
                    f"train_conf_mat_row_norm_{epoch}.png",
                )
                mlflow.log_figure(
                    train_conf_mat_col_norm_fig.figure_,
                    f"train_conf_mat_col_norm_{epoch}.png",
                )
                mlflow.log_figure(
                    val_conf_mat_row_norm_fig.figure_,
                    f"val_conf_mat_row_norm_{epoch}.png",
                )
                mlflow.log_figure(
                    val_conf_mat_col_norm_fig.figure_,
                    f"val_conf_mat_col_norm_{epoch}.png",
                )

                # close all the confusion matrix figures
                plt.close(train_conf_mat_row_norm_fig.figure_)
                plt.close(train_conf_mat_col_norm_fig.figure_)
                plt.close(val_conf_mat_row_norm_fig.figure_)
                plt.close(val_conf_mat_col_norm_fig.figure_)

                if checkpoint_type == "mlflow_api":
                    example_input, _ = next(iter(val_loader))
                    example_input.to(device)
                    example_output = model(example_input)
                    model_signature = infer_signature(
                        example_input,
                        example_output,
                    )

                    mlflow.pytorch.log_model(
                        model,
                        artifact_path=f"{model_name}_{epoch}",
                        signature=model_signature,
                    )
                else:
                    file_model_name = path_ckpt_models / f"{model_name}_{epoch}.pth"
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "model_class_name": model.__class__.__name__,
                            "model_config": {
                                "num_input_bands": num_input_bands,
                                "num_classes": num_classes,
                                "list_filters": list_filters,
                                "dropout_ratio": dropout_ratio,
                            },
                        },
                        file_model_name,
                    )
                    mlflow.log_artifact(
                        local_path=file_model_name, artifact_path="model_artifacts"
                    )
    logging.info("Training the EuroSAT MSI classification model complete!!!!")

    if checkpoint_type == "torch_api":
        shutil.rmtree(path_ckpt_models)
    return
