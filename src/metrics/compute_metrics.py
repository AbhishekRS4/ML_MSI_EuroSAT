import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from typing import List, Tuple, Union, Dict
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def compute_base_metrics(
    true_labels: Tensor,
    pred_labels: Tensor,
    average: str = "weighted",
) -> Tuple[float, float, float, float]:
    """
    compute base metrics

    ---------
    Arguments
    ---------
    true_labels: Tensor
        a torch tensor of true labels
    pred_labels: Tensor
        a torch tensor of predicted labels
    average: str
        a string indicating the kind of averaging that needs to be performed for multi-class scenario (default: weighted)

    -------
    Returns
    -------
    (acc_sc, f1_sc, pre_sc, rec_sc): Tuple[float, float, float, float]
        a tuple of base metrics like accuracy, f1, precision, recall
    """
    true_labels = true_labels.clone().detach().cpu().numpy()
    pred_labels = pred_labels.clone().detach().cpu().numpy()

    true_labels = true_labels.reshape(-1)
    pred_labels = pred_labels.reshape(-1)

    acc_sc = accuracy_score(true_labels, pred_labels)
    f1_sc = f1_score(true_labels, pred_labels, average=average)
    pre_sc = precision_score(true_labels, pred_labels, average=average)
    rec_sc = recall_score(true_labels, pred_labels, average=average)

    return acc_sc, f1_sc, pre_sc, rec_sc


def compute_confusion_matrix(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    list_label_names: np.ndarray,
    cmap: str = "Blues",
) -> Tuple[np.ndarray, ConfusionMatrixDisplay]:
    """
    compute confusion matrix

    ---------
    Arguments
    ---------
    true_labels: np.ndarray
        a torch tensor of true labels
    pred_labels: np.ndarray
        a torch tensor of predicted labels
    list_label_names: np.ndarray
        a numpy array of class label names
    cmap: str
        a string indicating the cmap to be used in the confusion matrix plot (default: cmap)

    -------
    Returns
    -------
    (conf_matrix, conf_matrix_fig): Tuple[np.ndarray, ConfusionMatrixDisplay]
        a tuple of numpy array of confusion matrix and a plot of confusion matrix figure
    """
    conf_matrix = confusion_matrix(
        true_labels,
        pred_labels,
        labels=np.arange(len(list_label_names)),
        normalize="true",
    )

    conf_matrix_fig = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=list_label_names
    )
    fig, ax = plt.subplots(figsize=(12, 12))
    conf_matrix_fig.plot(cmap=cmap, xticks_rotation="vertical", ax=ax)
    return conf_matrix, conf_matrix_fig


def compute_classification_report(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    list_label_names: np.ndarray,
) -> str:
    """
    compute classification report

    ---------
    Arguments
    ---------
    true_labels: np.ndarray
        a torch tensor of true labels
    pred_labels: np.ndarray
        a torch tensor of predicted labels
    list_label_names: np.ndarray
        a numpy array of class label names

    -------
    Returns
    -------
    clf_report: str
        a string of classificaition report
    """
    clf_report = classification_report(
        true_labels,
        pred_labels,
        labels=np.arange(len(list_label_names)),
        target_names=list_label_names,
    )

    return clf_report


def compute_additional_metrics(
    true_labels: Tensor, pred_labels: Tensor, list_label_names: np.ndarray
) -> Tuple[ConfusionMatrixDisplay, str]:
    """
    compute additional metrics for logging

    ---------
    Arguments
    ---------
    true_labels: Tensor
        a torch tensor of true labels
    pred_labels: Tensor
        a torch tensor of predicted labels
    list_label_names: np.ndarray
        a numpy array of class label names

    -------
    Returns
    -------
    conf_matrix_fig, clf_report: Tuple[ConfusionMatrixDisplay, str]
        a tuple of confusion matrix plot and a string of classification report
    """
    true_labels = true_labels.view(-1).clone().detach().cpu().numpy()
    pred_labels = pred_labels.view(-1).clone().detach().cpu().numpy()

    _, conf_matrix_fig = compute_confusion_matrix(
        true_labels, pred_labels, list_label_names
    )
    clf_report = compute_classification_report(
        true_labels, pred_labels, list_label_names
    )

    return conf_matrix_fig, clf_report
