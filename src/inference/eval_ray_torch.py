import logging
import torch
import numpy as np

import ray
import ray.train

from pathlib import Path
from typing import Dict, List, Tuple

from data_handler.data_bands import get_band_indices
from data_handler.ray_hf_streaming_dataset import (
    DATASET_PATH,
    EUROSAT_CLASS_NAMES,
    ray_preprocess,
    process_batch,
    get_ray_datasets_4_testing,
)
from metrics.compute_metrics import MetricsCalculator, get_confusion_matrix_figure
from models.msi_supervised import (
    MSI_ResNet,
    MSI_ResKANet,
    MSI_SE_ResNet,
    MSI_SE_ResKANet,
    MSI_PSA_ResNet,
    MSI_PSA_ResKANet,
)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_compile: bool = False,
) -> torch.nn.Module:
    """
    Load a trained model from a checkpoint file. Handles state dicts saved
    from both compiled (torch.compile) and non-compiled models.

    ---------
    Arguments
    ---------
    checkpoint_path: str
        full path to the model checkpoint (.pt or .pth file)
    device: torch.device
        torch device to load the model onto
    model_compile: bool
        whether to compile the loaded model with torch.compile (default: False)

    -------
    Returns
    -------
    model: torch.nn.Module
        the loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint["model_config"]
    model_name = model_config["model_name"]
    num_input_bands = model_config["num_input_bands"]
    num_classes = model_config["num_classes"]
    list_filters = model_config["list_filters"]
    dropout_ratio = model_config["dropout_ratio"]

    model_map = {
        "resnet": MSI_ResNet,
        "reskanet": MSI_ResKANet,
        "se_resnet": MSI_SE_ResNet,
        "se_reskanet": MSI_SE_ResKANet,
        "psa_resnet": MSI_PSA_ResNet,
        "psa_reskanet": MSI_PSA_ResKANet,
    }

    if model_name not in model_map:
        raise ValueError(f"Unidentified model name in checkpoint: {model_name}")

    model = model_map[model_name](
        num_input_bands=num_input_bands,
        num_classes=num_classes,
        list_filters=list_filters,
        dropout_ratio=dropout_ratio,
    )

    # Handle state dicts saved from compiled models.
    # torch.compile wraps the model and prefixes keys with "_orig_mod."
    state_dict = checkpoint["model_state_dict"]
    compiled_prefix = "_orig_mod."
    if any(k.startswith(compiled_prefix) for k in state_dict.keys()):
        state_dict = {
            k.removeprefix(compiled_prefix): v for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.to(device)

    if model_compile:
        model = torch.compile(model, mode="reduce-overhead")

    model.eval()
    return model


def evaluate_model(
    checkpoint_path: str,
    data_bands: List[str] = ["B", "G", "R", "NIR"],
    batch_size: int = 64,
    dataset_path: str = DATASET_PATH,
    output_log_file: str = "eval_ray_torch.log",
) -> None:
    """
    Evaluate the model performance on the validation and test datasets
    using Ray Data for streaming the HuggingFace parquet dataset.

    ---------
    Arguments
    ---------
    checkpoint_path: str
        full path to the model checkpoint (.pt or .pth file)
    data_bands: List[str]
        sentinel-2 bands to use for evaluation (default: [B, G, R, NIR])
    batch_size: int
        batch size for evaluation (default: 64)
    dataset_path: str
        HuggingFace dataset path (hf:// protocol)
    output_log_file: str
        file name for the output log file (default: eval_ray_torch.log)

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
        logging.warning("CUDA device not found, using CPU for evaluation.")
        device = torch.device("cpu")

    # Load model from checkpoint
    logging.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, device)

    logging.info(f"Data bands: {data_bands}")

    list_band_indices = get_band_indices(data_bands)
    num_classes = len(EUROSAT_CLASS_NAMES)

    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "working_dir": str(Path(__file__).resolve().parent.parent),
            }
        )

    # Create Ray Datasets for validation and test
    logging.info(f"Loading validation and test datasets from: {dataset_path}")
    val_ds, test_ds = get_ray_datasets_4_testing(dataset_path)

    # Preprocess datasets
    val_ds = val_ds.map_batches(
        ray_preprocess,
        fn_args=(list_band_indices,),
        batch_format="numpy",
        batch_size=batch_size,
        num_cpus=1,
        compute=ray.data.TaskPoolStrategy(size=4),
    )
    test_ds = test_ds.map_batches(
        ray_preprocess,
        fn_args=(list_band_indices,),
        batch_format="numpy",
        batch_size=batch_size,
        num_cpus=1,
        compute=ray.data.TaskPoolStrategy(size=4),
    )

    results = {}

    # Evaluate on both splits
    for split_name, dataset in [("validation", val_ds), ("test", test_ds)]:
        logging.info(f"Evaluating on {split_name} set...")

        metrics_calculator = MetricsCalculator(device, num_classes=num_classes)
        metrics_calculator.reset_metrics()

        running_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataset.iter_batches(
                batch_size=batch_size,
                batch_format="numpy",
                drop_last=False,
            ):
                images_tensor, labels_tensor = process_batch(batch, is_train=False)
                images_tensor = images_tensor.to(device, dtype=torch.float)
                labels_tensor = labels_tensor.to(device, dtype=torch.long)

                pred_logits = model(images_tensor)
                loss = torch.nn.functional.cross_entropy(pred_logits, labels_tensor)
                pred_labels = torch.argmax(pred_logits, dim=1)

                metrics_calculator.update_metrics(labels_tensor, pred_labels)
                running_loss += loss.item()
                num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)

        (
            acc,
            f1,
            prec,
            rec,
            conf_mat_row_norm,
            conf_mat_col_norm,
        ) = metrics_calculator.compute_metrics()

        split_results = {
            "loss": avg_loss,
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
        }
        results[split_name] = split_results

        logging.info(
            f"{split_name.capitalize()} - loss: {avg_loss:.4f}, "
            f"accuracy: {float(acc):.4f}, f1: {float(f1):.4f}, "
            f"precision: {float(prec):.4f}, recall: {float(rec):.4f}"
        )

    logging.info("Evaluation complete!")
    return
