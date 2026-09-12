import torch


def preprocess_msi_image(
    msi_image: torch.Tensor, threshold: int = pow(2, 14)
) -> torch.Tensor:
    """
    ---------
    Arguments
    ---------
    msi_image: torch.Tensor
        a torch tensor of raw MSI with the selected bands
    threshold: int
        threshold that needs to be used for preprocessing

    -------
    Returns
    -------
    msi_image: torch.Tensor
        a torch Tensor of preprocessed MSI with the selected bands
    """
    msi_image = torch.clamp(msi_image, 0, threshold)
    msi_image = msi_image / threshold
    return msi_image
