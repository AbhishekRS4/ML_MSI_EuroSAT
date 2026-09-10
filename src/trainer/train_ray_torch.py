import os
import time
import torch
import logging
import mlflow

import ray
import ray.train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig

from pathlib import Path
from typing import Dict, List
from torch import GradScaler
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import PolynomialLR

from loss_func.focal_loss import FocalLoss
from data_handler.data_bands import get_band_indices
from data_handler.ray_hf_streaming_dataset import (
    DATASET_PATH,
    EUROSAT_CLASS_NAMES,
    ray_preprocess,
    process_batch,
    get_ray_datasets_4_training,
)
from metrics.compute_metrics import MetricsCalculator
from models.msi_supervised import (
    MSI_ResNet,
    MSI_ResKANet,
    MSI_SE_ResNet,
    MSI_SE_ResKANet,
    MSI_PSA_ResNet,
    MSI_PSA_ResKANet,
)

os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_P2P_DISABLE"] = "1"


def get_model(
    model_name: str,
    num_input_bands: int,
    num_classes: int,
    list_filters: List[int],
    dropout_ratio: float,
):
    """
    Get the model based on the model name.

    ---------
    Arguments
    ---------
    model_name: str
        name of the model
    num_input_bands: int
        number of input bands
    num_classes: int
        number of output classes
    list_filters: List[int]
        list of filters for the model
    dropout_ratio: float
        dropout ratio

    -------
    Returns
    -------
    model: nn.Module
        the instantiated model
    """
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
        raise ValueError(f"Unidentified option for model_name: {model_name}")
    return model


def train_loop_per_worker(config: Dict):
    """
    Training loop that runs on each Ray Train worker.
    Uses Ray Data for streaming the HuggingFace parquet dataset.

    ---------
    Arguments
    ---------
    config: Dict
        configuration dictionary with training hyperparameters
    """
    # Extract config
    model_name = config["model_name"]
    num_input_bands = config["num_input_bands"]
    num_classes = config["num_classes"]
    list_filters = config["list_filters"]
    dropout_ratio = config["dropout_ratio"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    loss_fn = config["loss_fn"]
    data_bands = config["data_bands"]
    model_compile = config.get("model_compile", False)
    exp_name = config.get("exp_name", "eurosat_msi_ray_train")
    path_ckpt_models = Path(config.get("path_ckpt_models", "tmp_models"))
    out_log_file = config.get("out_log_file")

    # Determine if this is the rank-0 worker (only rank-0 should log to file)
    is_rank_zero = ray.train.get_context().get_world_rank() == 0
    device = ray.train.torch.get_device()

    # Create checkpoint directory in the worker's environment
    if is_rank_zero and not path_ckpt_models.is_dir():
        path_ckpt_models.mkdir(parents=True, exist_ok=True)

    # Setup logging - use a dedicated logger with explicit file handler
    logger = logging.getLogger("ray_trainer")
    logger.setLevel(logging.INFO if is_rank_zero else logging.WARNING)
    logger.handlers.clear()

    if is_rank_zero and out_log_file:
        # Use absolute path to ensure file is written to the correct location
        log_path = Path(out_log_file).resolve()
        file_handler = logging.FileHandler(str(log_path), mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(console_handler)

    # Setup MLflow only on rank-0 worker to avoid duplicate logging
    if is_rank_zero:
        mlflow_tracking_uri = config.get("mlflow_tracking_uri", None)
        if not mlflow_tracking_uri:
            # Default to an absolute path next to the checkpoint directory
            # so mlruns is created in the project directory, not Ray's temp dir
            mlflow_tracking_uri = f"file://{path_ckpt_models.parent / 'mlruns'}"
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        mlflow.set_experiment(exp_name)
        mlflow.start_run()

        # Log parameters
        mlflow.log_param("optimization.optimizer_name", "adamw")
        mlflow.log_param("optimization.num_epochs", num_epochs)
        mlflow.log_param("optimization.learning_rate", learning_rate)
        mlflow.log_param("optimization.weight_decay", weight_decay)
        mlflow.log_param("optimization.batch_size", batch_size)
        mlflow.log_param("optimization.loss_fn", loss_fn)

        mlflow.log_param("dataset.data_bands", data_bands)
        mlflow.log_param("dataset.num_classes", num_classes)
        mlflow.log_param("dataset.dataset_path", DATASET_PATH)
        mlflow.log_param("dataset.class_names", EUROSAT_CLASS_NAMES)

        mlflow.log_param("model.model_name", model_name)
        mlflow.log_param("model.list_filters", list_filters)
        mlflow.log_param("model.num_input_bands", num_input_bands)
        mlflow.log_param("model.dropout_ratio", dropout_ratio)
        mlflow.log_param("model.model_compile", model_compile)

    # Build model
    model = get_model(
        model_name, num_input_bands, num_classes, list_filters, dropout_ratio
    )

    # Optionally compile the model for reduced overhead
    if model_compile:
        model = torch.compile(model, mode="reduce-overhead")

    # Prepare model for distributed training
    model = ray.train.torch.prepare_model(model)

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    lr_scheduler = PolynomialLR(optimizer, num_epochs + 1, power=0.95)

    # Setup loss function
    if loss_fn == "cross_entropy":
        criterion = CrossEntropyLoss()
    elif loss_fn == "focal":
        criterion = FocalLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    # Setup AMP scaler
    amp_scaler = GradScaler(device="cuda")

    # Metrics calculators (sync across workers in distributed training)
    train_metrics = MetricsCalculator(
        device,
        num_classes=num_classes,
    )
    val_metrics = MetricsCalculator(
        device,
        num_classes=num_classes,
    )

    # Get the Ray Data shards for this worker
    train_data_shard = ray.train.get_dataset_shard("train")
    val_data_shard = ray.train.get_dataset_shard("validation")

    best_val_acc = 0.0

    if is_rank_zero:
        logger.info(
            f"Training started - model: {model_name}, num_classes: {num_classes}, "
            f"bands: {data_bands}, epochs: {num_epochs}"
        )

    for epoch in range(1, num_epochs + 1):
        train_metrics.reset_metrics()
        val_metrics.reset_metrics()

        time_start = time.time()

        # ---- Training ----
        model.train()
        running_train_loss = 0.0
        num_train_batches = 0

        train_batch_iter = train_data_shard.iter_batches(
            batch_size=batch_size,
            batch_format="numpy",
            prefetch_batches=4,
            local_shuffle_buffer_size=batch_size * 8,
            drop_last=True,
        )

        for batch in train_batch_iter:
            images_tensor, labels_tensor = process_batch(batch, is_train=True)
            images_tensor = images_tensor.to(device, dtype=torch.float)
            labels_tensor = labels_tensor.to(device, dtype=torch.long)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred_logits = model(images_tensor)
                loss = criterion(pred_logits, labels_tensor)

            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

            pred_labels = torch.argmax(pred_logits, dim=1)
            train_metrics.update_metrics(labels_tensor, pred_labels)
            running_train_loss += loss.item()
            num_train_batches += 1

        train_loss = running_train_loss / max(num_train_batches, 1)

        (
            train_acc,
            train_f1,
            train_prec,
            train_rec,
            _,
            _,
        ) = train_metrics.compute_metrics()

        # ---- Validation ----
        model.eval()
        running_val_loss = 0.0
        num_val_batches = 0

        val_batch_iter = val_data_shard.iter_batches(
            batch_size=batch_size,
            batch_format="numpy",
            drop_last=True,
        )

        with torch.no_grad():
            for batch in val_batch_iter:
                images_tensor, labels_tensor = process_batch(batch, is_train=False)
                images_tensor = images_tensor.to(device, dtype=torch.float)
                labels_tensor = labels_tensor.to(device, dtype=torch.long)

                pred_logits = model(images_tensor)
                loss = criterion(pred_logits, labels_tensor)
                pred_labels = torch.argmax(pred_logits, dim=1)

                val_metrics.update_metrics(labels_tensor, pred_labels)
                running_val_loss += loss.item()
                num_val_batches += 1

        val_loss = running_val_loss / max(num_val_batches, 1)

        (
            val_acc,
            val_f1,
            val_prec,
            val_rec,
            _,
            _,
        ) = val_metrics.compute_metrics()

        lr_scheduler.step()
        time_end = time.time()

        # Log epoch metrics to file (only from rank-0 to avoid duplicate logs)
        if is_rank_zero:
            logger.info(
                f"Epoch: {epoch}/{num_epochs}, time: {time_end - time_start:.4f} sec."
            )
            logger.info(
                f"  Train - loss: {train_loss:.4f}, accuracy: {float(train_acc):.4f}, "
                f"f1: {float(train_f1):.4f}, precision: {float(train_prec):.4f}, "
                f"recall: {float(train_rec):.4f}, num train batches: {num_train_batches}"
            )
            logger.info(
                f"  Val   - loss: {val_loss:.4f}, accuracy: {float(val_acc):.4f}, "
                f"f1: {float(val_f1):.4f}, precision: {float(val_prec):.4f}, "
                f"recall: {float(val_rec):.4f}, num val batches: {num_val_batches}"
            )
            logger.info(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Log metrics to MLflow (only from rank-0)
        if is_rank_zero:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": float(train_acc),
                    "train_f1": float(train_f1),
                    "train_precision": float(train_prec),
                    "train_recall": float(train_rec),
                    "val_loss": val_loss,
                    "val_accuracy": float(val_acc),
                    "val_f1": float(val_f1),
                    "val_precision": float(val_prec),
                    "val_recall": float(val_rec),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_time_sec": time_end - time_start,
                },
                step=epoch,
            )

        # Report metrics to Ray Train (MUST be called by ALL workers)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": float(train_acc),
            "train_f1": float(train_f1),
            "train_precision": float(train_prec),
            "train_recall": float(train_rec),
            "val_loss": val_loss,
            "val_accuracy": float(val_acc),
            "val_f1": float(val_f1),
            "val_precision": float(val_prec),
            "val_recall": float(val_rec),
            "epoch_time_sec": time_end - time_start,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        if is_rank_zero:
            # Save checkpoint if validation accuracy improved
            if float(val_acc) >= best_val_acc:
                best_val_acc = float(val_acc)
                checkpoint_data = {
                    "model_state_dict": (
                        model.module.state_dict()
                        if hasattr(model, "module")
                        else model.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "model_config": {
                        "model_name": model_name,
                        "num_input_bands": num_input_bands,
                        "num_classes": num_classes,
                        "list_filters": list_filters,
                        "dropout_ratio": dropout_ratio,
                    },
                }
                file_model_name = path_ckpt_models / f"{model_name}_{epoch}.pt"
                torch.save(
                    checkpoint_data,
                    file_model_name,
                )
                mlflow.log_artifact(
                    local_path=str(file_model_name), artifact_path="model_artifacts"
                )
                file_model_name.unlink()

        # ray.train.report must be called by all workers (collective operation)
        ray.train.report(metrics=metrics)

    if is_rank_zero:
        logger.info(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
        mlflow.end_run()

    return


def train_pipeline_ray(
    model_name: str = "resnet",
    list_filters: List[int] = [64, 128, 256],
    dropout_ratio: float = 0.2,
    loss_fn: str = "focal",
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-5,
    batch_size: int = 64,
    data_bands: List[str] = ["B", "G", "R", "NIR"],
    num_workers: int = 1,
    use_gpu: bool = True,
    num_cpu_workers: int = 4,
    model_compile: bool = False,
    dataset_path: str = DATASET_PATH,
    exp_name: str = "eurosat_msi_ray_train",
    mlflow_tracking_uri: str = None,
    run_name: str = "eurosat_msi_ray_train",
    out_log_file: str = "ray_torch_trainer.log",
) -> None:
    """
    Main training pipeline using Ray TorchTrainer with streaming
    HuggingFace parquet dataset.

    ---------
    Arguments
    ---------
    model_name: str
        model name (default: resnet)
    list_filters: List[int]
        list of filters for the model (default: [64, 128, 256])
    dropout_ratio: float
        dropout ratio (default: 0.2)
    loss_fn: str
        loss function name (default: focal)
    num_epochs: int
        number of training epochs (default: 100)
    learning_rate: float
        learning rate (default: 1e-3)
    weight_decay: float
        weight decay (default: 5e-5)
    batch_size: int
        batch size per worker (default: 64)
    data_bands: List[str]
        sentinel-2 bands to use (default: [B, G, R, NIR])
    num_workers: int
        number of Ray Train workers (default: 1)
    use_gpu: bool
        whether to use GPU (default: True)
    model_compile: bool
        whether to compile the model (default: False)
    dataset_path: str
        HuggingFace dataset path
    exp_name: str
        MLflow experiment name (default: eurosat_msi_ray_train)
    mlflow_tracking_uri: str
        MLflow tracking URI for remote server (default: None, uses local)
    run_name: str
        name for the Ray Train run

    -------
    Returns
    -------
    result: ray.train.Result
        the training result from Ray Train
    """
    dir_ckpt_models = "tmp_models"
    path_ckpt_models = Path(dir_ckpt_models).resolve()

    if not path_ckpt_models.is_dir():
        path_ckpt_models.mkdir(parents=True, exist_ok=True)

    num_input_bands = len(data_bands)
    num_classes = len(EUROSAT_CLASS_NAMES)
    list_band_indices = get_band_indices(data_bands)

    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "working_dir": str(Path(__file__).resolve().parent.parent),
            }
        )

    # Create Ray Datasets from HuggingFace parquet files
    logging.info(f"Loading datasets from: {dataset_path}")
    train_ds, val_ds = get_ray_datasets_4_training(dataset_path)
    train_ds = train_ds.repartition(num_blocks=16).map_batches(
        ray_preprocess,
        fn_args=(list_band_indices,),
        batch_format="numpy",
        batch_size=batch_size,
        num_cpus=1,
        compute=ray.data.TaskPoolStrategy(size=4),
    )
    val_ds = val_ds.repartition(num_blocks=16).map_batches(
        ray_preprocess,
        fn_args=(list_band_indices,),
        batch_format="numpy",
        batch_size=batch_size,
        num_cpus=1,
        compute=ray.data.TaskPoolStrategy(size=4),
    )

    logging.info(f"Train dataset schema: {train_ds.schema()}")

    # Training config passed to each worker
    train_config = {
        "model_name": model_name,
        "num_input_bands": num_input_bands,
        "num_classes": num_classes,
        "list_filters": list_filters,
        "dropout_ratio": dropout_ratio,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "loss_fn": loss_fn,
        "data_bands": data_bands,
        "model_compile": model_compile,
        "exp_name": exp_name,
        "mlflow_tracking_uri": mlflow_tracking_uri,
        "path_ckpt_models": str(path_ckpt_models),
        "out_log_file": str(Path(out_log_file).resolve()),
    }

    # Scaling config for distributed training
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"CPU": num_cpu_workers, "GPU": 1 if use_gpu else 0},
    )

    # Run config
    run_config = RunConfig(
        name=run_name,
    )

    # Create the TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={
            "train": train_ds.randomize_block_order(),
            "validation": val_ds,
        },
    )

    # Run training
    logging.info(f"Starting Ray TorchTrainer with {num_workers} worker(s)...")
    logging.info(f"Model: {model_name}, Bands: {data_bands}, Epochs: {num_epochs}")
    result = trainer.fit()

    logging.info("Training complete!")

    return
