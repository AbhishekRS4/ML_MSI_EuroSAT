from typing import Tuple, List
from pathlib import Path, PosixPath


def get_list_files_n_labels(
    dir_dataset: str,
) -> Tuple[List[PosixPath], List[int], List[str]]:
    """
    get a list of files, corresponding labels and class instances

    ---------
    Arguments
    ---------
    dir_dataset: str
        a list of train image files

    -------
    Returns
    -------
    (list_images, list_labels, list_class_dirs) : Tuple[List[PosixPath], List[int], List[str]]
        a tuple of list of image files, corresponding labels and class instances

    """
    path_dataset = Path(dir_dataset)

    list_class_dirs = sorted([d.name for d in path_dataset.glob("*") if d.is_dir()])
    list_images = []
    list_labels = []

    for class_idx, class_name in enumerate(list_class_dirs):
        path_class = path_dataset / class_name
        class_list_images = [f for f in path_class.glob("*.tif") if f.is_file()]

        list_images = list_images + class_list_images
        list_labels = list_labels + [class_idx] * len(class_list_images)

    return list_images, list_labels, list_class_dirs
