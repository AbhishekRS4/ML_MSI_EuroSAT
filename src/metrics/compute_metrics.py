import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from typing import List, Tuple, Union
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix


class MetricsCalculator:
    def __init__(
        self,
        device: torch.device,
        task: str = "multiclass",
        num_classes: int = 10,
        average: str = "weighted",
        dist_sync_on_step: bool = False,
        sync_on_compute: bool = True,
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
        dist_sync_on_step: bool
            whether to synchronize metric states across processes at each step.
            Set to True for distributed training to aggregate metrics across all
            workers before computing. (default: False)
        """
        self.task = task
        self.device = device
        self.average = average
        self.num_classes = num_classes
        self.dist_sync_on_step = dist_sync_on_step
        self.sync_on_compute = sync_on_compute

        self.accuracy_score = Accuracy(
            task=self.task,
            num_classes=self.num_classes,
            dist_sync_on_step=self.dist_sync_on_step,
            sync_on_compute=self.sync_on_compute,
        ).to(self.device)
        self.f1_score = F1Score(
            task=self.task,
            num_classes=self.num_classes,
            average=self.average,
            dist_sync_on_step=self.dist_sync_on_step,
            sync_on_compute=self.sync_on_compute,
        ).to(self.device)
        self.precision_score = Precision(
            task=self.task,
            num_classes=self.num_classes,
            average=self.average,
            dist_sync_on_step=self.dist_sync_on_step,
            sync_on_compute=self.sync_on_compute,
        ).to(self.device)
        self.recall_score = Recall(
            task=self.task,
            num_classes=self.num_classes,
            average=self.average,
            dist_sync_on_step=self.dist_sync_on_step,
            sync_on_compute=self.sync_on_compute,
        ).to(self.device)
        self.conf_matrix_row_norm = ConfusionMatrix(
            task=self.task,
            num_classes=self.num_classes,
            normalize="true",
            dist_sync_on_step=self.dist_sync_on_step,
            sync_on_compute=self.sync_on_compute,
        ).to(self.device)
        self.conf_matrix_col_norm = ConfusionMatrix(
            task=self.task,
            num_classes=self.num_classes,
            normalize="pred",
            dist_sync_on_step=self.dist_sync_on_step,
            sync_on_compute=self.sync_on_compute,
        ).to(self.device)

    def update_metrics(
        self,
        true_labels: Tensor,
        pred_labels: Tensor,
    ) -> None:
        """
        update the metrics

        ---------
        Arguments
        ---------
        true_labels: Tensor
            a torch tensor of true labels
        pred_labels: Tensor
            a torch tensor of predicted labels
        """
        true_labels = true_labels.view(-1)
        pred_labels = pred_labels.view(-1)

        self.accuracy_score.update(pred_labels, true_labels)
        self.f1_score.update(pred_labels, true_labels)
        self.precision_score.update(pred_labels, true_labels)
        self.recall_score.update(pred_labels, true_labels)
        self.conf_matrix_row_norm.update(pred_labels, true_labels)
        self.conf_matrix_col_norm.update(pred_labels, true_labels)
        return

    def compute_metrics(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        compute the metrics

        -------
        Returns
        -------
        (acc_sc, f1_sc, pre_sc, rec_sc, conf_matrix_row_norm, conf_matrix_col_norm):
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
            a tuple of base metrics like accuracy, f1, precision, recall
        """
        return (
            self.accuracy_score.compute(),
            self.f1_score.compute(),
            self.precision_score.compute(),
            self.recall_score.compute(),
            self.conf_matrix_row_norm.compute(),
            self.conf_matrix_col_norm.compute(),
        )

    def reset_metrics(self) -> None:
        """
        reset the metrics
        """
        self.accuracy_score.reset()
        self.f1_score.reset()
        self.precision_score.reset()
        self.recall_score.reset()
        self.conf_matrix_row_norm.reset()
        self.conf_matrix_col_norm.reset()
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
