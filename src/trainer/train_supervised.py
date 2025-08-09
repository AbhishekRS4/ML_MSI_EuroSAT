import sys
import time
import torch
import mlflow
import logging
import numpy as np
import torch.nn.functional as F

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
from metrics.compute_metrics import compute_base_metrics, compute_additional_metrics
from models.msi_supervised import (
    MSI_ResNet,
    MSI_ResKANet,
    MSI_SE_ResNet,
    MSI_SE_ResKANet,
)


def train(
    model: Union[
        MSI_ResNet,
        MSI_ResKANet,
        MSI_SE_ResNet,
        MSI_SE_ResKANet,
    ],
    optimizer: AdamW,
    criterion: CrossEntropyLoss,
    train_loader: DataLoader,
    device: torch.device,
    scaler: GradScaler,
) -> float:
    """
    train function

    ---------
    Arguments
    ---------
    model: Union[MSI_ResNet, MSI_ResKANet,]
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

    -------
    Returns
    -------
    train_loss: float
        training loss
    """
    model.to(device)
    model.train()
    train_running_loss = 0.0
    num_train_batches = len(train_loader)

    for msi_bands, gt_labels in train_loader:
        msi_bands = msi_bands.to(device, dtype=torch.float)
        gt_labels = gt_labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        # run forward pass with autocast
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred_logits = model(msi_bands)
            loss = criterion(pred_logits, gt_labels)
            train_running_loss += loss.item()

        # loss is scaled and then scaled gradients are created
        scaler.scale(loss).backward()
        # apply the update
        scaler.step(optimizer)
        # update the scaler for the next iteration
        scaler.update()

    train_loss = train_running_loss / num_train_batches
    return train_loss


def validate(
    model: Union[
        MSI_ResNet,
        MSI_ResKANet,
        MSI_SE_ResNet,
        MSI_SE_ResKANet,
    ],
    criterion: CrossEntropyLoss,
    validation_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    validation function

    ---------
    Arguments
    ---------
    model: Union[MSI_ResNet, MSI_ResKANet,]
        an object of type torch model
    criterion: CrossEntropyLoss
        an object of type torch criterion function
    validation_loader: DataLoader
        an object of type torch dataloader
    device: torch.device
        an object of type torch device

    -------
    Returns
    -------
    validation_loss: float
        validation loss
    """
    model.to(device)
    model.eval()
    validation_running_loss = 0.0
    num_validation_batches = len(validation_loader)

    with torch.no_grad():
        for msi_bands, gt_labels in validation_loader:
            msi_bands = msi_bands.to(device, dtype=torch.float)
            gt_labels = gt_labels.to(device, dtype=torch.long)

            pred_logits = model(msi_bands)
            loss = criterion(pred_logits, gt_labels)

            validation_running_loss += loss.item()

    validation_loss = validation_running_loss / num_validation_batches
    return validation_loss


def predict_n_compute_metrics(
    model: Union[
        MSI_ResNet,
        MSI_ResKANet,
        MSI_SE_ResNet,
        MSI_SE_ResKANet,
    ],
    test_loader: DataLoader,
    device: torch.device,
    list_label_names: np.ndarray,
) -> Tuple[float, float, float, float, ConfusionMatrixDisplay, str]:
    """
    predict and compute additional metrics for logging

    ---------
    Arguments
    ---------
    model: Union[MSI_ResNet, MSI_ResKANet,]
        an object of type torch model
    test_loader: DataLoader
        an object of type torch dataloader
    device: torch.device
        an object of type torch device
    list_label_names: np.ndarray
        a numpy array of label names

    -------
    Returns
    -------
    acc_sc, f1_sc, pre_sc, rec_sc, conf_matrix_fig, clf_report: Tuple[float, float, float, float, ConfusionMatrixDisplay, str]
        a tuple of accuracy, f1, precision, recall, confusion matrix plot, and classification report
    """
    model.to(device)
    model.eval()

    all_true_labels, all_pred_labels = None, None

    with torch.no_grad():
        for msi_bands, gt_labels in test_loader:
            msi_bands = msi_bands.to(device, dtype=torch.float)
            gt_labels = gt_labels.to(device, dtype=torch.long)

            pred_logits = model(msi_bands)
            pred_labels = torch.argmax(pred_logits, dim=1)

            if all_true_labels is not None and all_pred_labels is not None:
                all_true_labels = torch.cat([all_true_labels, gt_labels])
                all_pred_labels = torch.cat([all_pred_labels, pred_labels])
            else:
                all_true_labels = gt_labels
                all_pred_labels = pred_labels

    acc_sc, f1_sc, pre_sc, rec_sc = compute_base_metrics(
        all_true_labels, all_pred_labels
    )
    conf_matrix_fig, clf_report = compute_additional_metrics(
        all_true_labels, all_pred_labels, list_label_names
    )

    return acc_sc, f1_sc, pre_sc, rec_sc, conf_matrix_fig, clf_report


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
    validation_size: float = 0.2,
    num_workers: int = 8,
    data_bands: List[str] = ["B", "G", "R"],
    output_log_file: str = "trainer.log",
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
    validation_size: float
        validation set size (default: 0.2)
    num_workers: int
        number of workers to be used for data loading (default: 8)
    data_bands: List[str]
        a list of data bands that needs to be used for training the model (default: [B, G, R])
    output_log_file: str
        file name for the output log file (default: trainer.log)
    """
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
    list_images, list_labels, list_class_names = get_list_files_n_labels(dir_dataset)
    num_classes = list_labels[-1] + 1

    num_data_samples = len(list_images)
    logging.info(f"Num data samples: {num_data_samples}")

    train_loader, validation_loader = get_dataloaders_for_training(
        list_images,
        list_labels,
        data_bands,
        validation_size=validation_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # logging.info(list_images[0:5], list_labels[0:5], list_class_names[0:5])

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
    else:
        logging.info(f"Unidentified option for arg (model_name): {model_name}")
    model.to(device)

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
    best_validation_acc = 0

    logging.info(
        f"Training the EuroSAT MSI classification model started, model_name: {model_name}, num_classes: {num_classes}"
    )

    best_validation_acc = 0

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

        for epoch in range(1, num_epochs + 1):
            time_start = time.time()
            train_loss = train(
                model, optimizer, criterion, train_loader, device, scaler
            )
            validation_loss = validate(model, criterion, validation_loader, device)

            (
                train_acc,
                train_f1,
                train_pre,
                train_rec,
                train_conf_matrix,
                train_clf_report,
            ) = predict_n_compute_metrics(model, train_loader, device, list_class_names)
            (
                validation_acc,
                validation_f1,
                validation_pre,
                validation_rec,
                validation_conf_matrix,
                validation_clf_report,
            ) = predict_n_compute_metrics(model, train_loader, device, list_class_names)
            time_end = time.time()
            logging.info(
                f"Epoch: {epoch}/{num_epochs}, time: {time_end-time_start:.4f} sec."
            )
            logging.info(
                f"Train set, loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, f1: {train_f1:.4f}, precision: {train_pre:.4f}, recall: {train_rec:.4f}"
            )
            logging.info(
                f"Validation set, loss: {validation_loss:.4f}, accuracy: {validation_acc:.4f}, f1: {validation_f1:.4f}, precision: {validation_pre:.4f}, recall: {validation_rec:.4f}\n"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("train_precision", train_pre, step=epoch)
            mlflow.log_metric("train_recall", train_rec, step=epoch)

            mlflow.log_metric("validation_loss", validation_loss, step=epoch)
            mlflow.log_metric("validation_accuracy", validation_acc, step=epoch)
            mlflow.log_metric("validation_f1", validation_f1, step=epoch)
            mlflow.log_metric("validation_precision", validation_pre, step=epoch)
            mlflow.log_metric("validation_recall", validation_rec, step=epoch)

            if validation_acc >= best_validation_acc:
                best_validation_acc = validation_acc
                mlflow.log_text(train_clf_report, f"train_clf_report_{epoch}.txt")
                mlflow.log_text(
                    validation_clf_report, f"validation_clf_report_{epoch}.txt"
                )

                mlflow.log_figure(
                    train_conf_matrix.figure_, f"train_conf_matrix_{epoch}.png"
                )
                mlflow.log_figure(
                    validation_conf_matrix.figure_,
                    f"validation_conf_matrix_{epoch}.png",
                )

                best_model = deepcopy(model)
                best_model.to("cpu")

                example_input, _ = next(iter(validation_loader))
                example_input.to("cpu")
                example_output = best_model(example_input)
                model_signature = infer_signature(
                    example_input.cpu().numpy(),
                    example_output.detach().cpu().numpy(),
                )

                mlflow.pytorch.log_model(
                    best_model,
                    artifact_path=f"{model_name}_{epoch}",
                    signature=model_signature,
                )

            lr_scheduler.step()
    logging.info("Training the EuroSAT MSI classification model complete!!!!")
    return
