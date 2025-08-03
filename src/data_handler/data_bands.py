from typing import List

SENTINEL2_BAND_INDICES = {
    "B": 2,
    "G": 3,
    "R": 4,
    "NIR": 8,
    "SWIR-1": 12,
    "SWIR-2": 13,
}


def get_band_indices(list_bands: List[str]) -> List[int]:
    """
    return data band indices for the sentinel-2 MSI data

    ---------
    Arguments
    ---------
    list_bands: List[str]
        a list of sentinel-2 MSI data bands

    -------
    Returns
    -------
    (band_indices): List[int]
        a list of data band indices
    """
    band_indices = []

    for band in list_bands:
        band_indices.append(SENTINEL2_BAND_INDICES[band])
    return band_indices
