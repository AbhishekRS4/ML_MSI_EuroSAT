import torch

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
        state_dict = {k.removeprefix(compiled_prefix): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)

    if model_compile:
        model = torch.compile(model, mode="reduce-overhead")

    model.eval()
    return model
