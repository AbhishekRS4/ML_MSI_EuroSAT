import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from typing import List, Tuple, Union, Dict
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix


class MetricsCalculator:
    def __init__(
        self,
        device: torch.device,
        task: str = "multiclass",
        num_classes: int = 10,
        average: str = "weighted",
    ):
        """
        MetricsCalculator class to compute some of the important metrics

        ----------
        Attributes
        ----------
        device: torch.device
            indicates the torch device type
        task: str
            a string indicating the task (default: multiclass)
        num_classes: int
            an integer with the number of classes
        average: str
            a string indicating the type of averaging that needs to be performed for multi-class scenario (default: weighted)
        """
        self.task = task
        self.device = device
        self.average = average
        self.num_classes = num_classes

        self.accuracy_scorer = Accuracy(
            task=self.task, num_classes=self.num_classes
        ).to(self.device)
        self.f1_scorer = F1Score(
            task=self.task, num_classes=self.num_classes, average=self.average
        ).to(self.device)
        self.precision_scorer = Precision(
            task=self.task, num_classes=self.num_classes, average=self.average
        ).to(self.device)
        self.recall_scorer = Recall(
            task=self.task, num_classes=self.num_classes, average=self.average
        ).to(self.device)
        self.conf_matrix_row_normalized = ConfusionMatrix(
            task=self.task,
            num_classes=self.num_classes,
            normalize="true",
        ).to(self.device)
        self.conf_matrix_col_normalized = ConfusionMatrix(
            task=self.task,
            num_classes=self.num_classes,
            normalize="pred",
        ).to(self.device)

    def compute_base_metrics(
        self,
        true_labels: Tensor,
        pred_labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        compute base metrics

        ---------
        Arguments
        ---------
        true_labels: Tensor
            a torch tensor of true labels
        pred_labels: Tensor
            a torch tensor of predicted labels

        -------
        Returns
        -------
        (acc_sc, f1_sc, pre_sc, rec_sc): Tuple[Tensor, Tensor, Tensor, Tensor]
            a tuple of base metrics like accuracy, f1, precision, recall
        """
        true_labels = true_labels.view(-1)
        pred_labels = pred_labels.view(-1)

        acc_sc = self.accuracy_scorer(pred_labels, true_labels)
        f1_sc = self.f1_scorer(pred_labels, true_labels)
        pre_sc = self.precision_scorer(pred_labels, true_labels)
        rec_sc = self.recall_scorer(pred_labels, true_labels)

        return acc_sc, f1_sc, pre_sc, rec_sc

    def update_confusion_matrix(
        self,
        true_labels: Tensor,
        pred_labels: Tensor,
    ) -> None:
        """
        update the confusion matrix

        ---------
        Arguments
        ---------
        true_labels: Tensor
            a torch tensor of true labels
        pred_labels: Tensor
            a torch tensor of predicted labels
        """
        self.conf_matrix_row_normalized.update(pred_labels, true_labels)
        self.conf_matrix_col_normalized.update(pred_labels, true_labels)
        return

    def compute_confusion_matrix(self) -> Tensor:
        """
        compute confusion matrix

        -------
        Returns
        -------
        conf_matrix: Tensor
            a tensor of confusion matrix
        """
        return (
            self.conf_matrix_row_normalized.compute(),
            self.conf_matrix_col_normalized.compute(),
        )

    def reset_confusion_matrix(self) -> None:
        """
        reset the confusion matrix
        """
        self.conf_matrix_row_normalized.reset()
        self.conf_matrix_col_normalized.reset()
        return


def get_confusion_matrix_figure(
    conf_matrix: np.ndarray,
    list_label_names: Union[List[str], np.ndarray],
    scale_to_percent: bool = True,
    cmap: str = "Blues",
) -> ConfusionMatrixDisplay:
    """
    get confusion matrix figure

    ---------
    Arguments
    ---------
    conf_matrix: np.ndarray
        a numpy array of confusion matrix
    list_label_names: Union[List[str] np.ndarray]
        a list or numpy array of class label names
    scale_to_percent: bool
        a boolean indicating whether to scale the confusion matrix to percentage (default: True)
    cmap: str
        a string indicating the cmap to be used in the confusion matrix plot (default: blues)

    -------
    Returns
    -------
    conf_matrix_fig: ConfusionMatrixDisplay
        a figure of confusion matrix
    """
    if scale_to_percent:
        conf_matrix *= 100

    conf_matrix_fig = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=list_label_names
    )
    fig, ax = plt.subplots(figsize=(12, 12))
    conf_matrix_fig.plot(cmap=cmap, xticks_rotation="vertical", ax=ax)
    return conf_matrix_fig
