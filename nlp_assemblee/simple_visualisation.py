from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torchview import draw_graph

sns.set_context("paper")
sns.set_palette("deep")
sns.set_style("white")

colors = sns.color_palette("deep")


def calculate_metrics(results):
    y_true = results["labels"]
    y_pred = results["predictions"]
    probs = results["probs"]

    metrics = {
        "log_loss": log_loss(y_true, probs),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "recall_micro": recall_score(y_true, y_pred, average="micro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "precision_micro": precision_score(y_true, y_pred, average="micro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "f1_score_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_score_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_score_macro": f1_score(y_true, y_pred, average="macro"),
        "AUC_weighted_ovr": roc_auc_score(y_true, probs, average="weighted", multi_class="ovr"),
        "AUC_macro_ovr": roc_auc_score(y_true, probs, average="macro", multi_class="ovr"),
        "AUC_weighted_ovo": roc_auc_score(y_true, probs, average="weighted", multi_class="ovo"),
        "AUC_macro_ovo": roc_auc_score(y_true, probs, average="macro", multi_class="ovo"),
        "jaccard_weighted": jaccard_score(y_true, y_pred, average="weighted"),
        "jaccard_micro": jaccard_score(y_true, y_pred, average="micro"),
        "jaccard_macro": jaccard_score(y_true, y_pred, average="macro"),
        "matthews_weighted": matthews_corrcoef(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "confusion_matrix_true_normed": confusion_matrix(y_true, y_pred, normalize="true").tolist(),
        "confusion_matrix_pred_normed": confusion_matrix(y_true, y_pred, normalize="pred").tolist(),
        "confusion_matrix_all_normed": confusion_matrix(y_true, y_pred, normalize="all").tolist(),
    }

    return metrics


def calculate_metrics_binary(results):
    y_true = results["labels"]
    y_pred = results["predictions"]
    probs = results["probs"][:, 1]

    metrics = {
        "log_loss": log_loss(y_true, probs),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, average="binary"),
        "precision": precision_score(y_true, y_pred, average="binary"),
        "f1_score": f1_score(y_true, y_pred, average="binary"),
        "AUC": roc_auc_score(y_true, probs),
        "jaccard_weighted": jaccard_score(y_true, y_pred, average="binary"),
        "matthews_weighted": matthews_corrcoef(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "confusion_matrix_true_normed": confusion_matrix(y_true, y_pred, normalize="true").tolist(),
        "confusion_matrix_pred_normed": confusion_matrix(y_true, y_pred, normalize="pred").tolist(),
        "confusion_matrix_all_normed": confusion_matrix(y_true, y_pred, normalize="all").tolist(),
    }

    return metrics


def plot_precision_recall_curve(results, figsize=(6, 6), palette="deep"):
    y_onehot_test = pd.get_dummies(results["labels"]).values
    y_score = results["probs"]

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette(palette)

    n_classes = 3

    precisions, recalls, f1s, thresholds = dict(), dict(), dict(), dict()
    precisions["micro"], recalls["micro"], thresholds["micro"] = precision_recall_curve(
        y_onehot_test.ravel(), y_score.ravel()
    )
    f1s["micro"] = (
        2 * (precisions["micro"] * recalls["micro"]) / (precisions["micro"] + recalls["micro"])
    )

    for i in range(n_classes):
        precisions[i], recalls[i], thresholds[i] = precision_recall_curve(
            y_onehot_test[:, i], y_score[:, i]
        )
        f1s[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])

    ax.plot([0] + list(thresholds["micro"]), f1s["micro"], color=colors[3], linestyle=":", lw=4)
    # ax.plot(thresholds["micro"], precisions["micro"][:-1],
    #  color=colors[3], linestyle="-.", alpha=0.7, lw=1.5)
    # ax.plot(thresholds["micro"], recalls["micro"][:-1],
    #  color=colors[3], linestyle="--", alpha=0.7, lw=1.5)

    colors = sns.color_palette("deep")
    for class_id, color in zip(range(n_classes), colors):
        ax.plot(
            [0] + list(thresholds[class_id]),
            f1s[class_id],
            color=color,
            linestyle="-",
            lw=1.5,
            alpha=0.85,
        )
        ax.plot(
            [0] + list(thresholds[class_id]),
            precisions[class_id],
            color=color,
            linestyle="-.",
            alpha=0.5,
            lw=1,
        )
        ax.plot(
            [0] + list(thresholds[class_id]),
            recalls[class_id],
            color=color,
            linestyle="--",
            alpha=0.5,
            lw=1,
        )

    colors_lines = [
        Line2D([0], [0], color=colors[0], lw=1),
        Line2D([0], [0], color=colors[1], lw=1),
        Line2D([0], [0], color=colors[2], lw=1),
        Line2D([0], [0], color=colors[3], lw=4, linestyle=":"),
    ]
    type_lines = [
        Line2D([0], [0], color="k", lw=1),
        Line2D([0], [0], color="k", lw=1, linestyle="-."),
        Line2D([0], [0], color="k", lw=1, linestyle="--"),
    ]

    colors_labels = ["Gauche", "Centre", "Droite", "Micro-average"]
    type_labels = ["F1-score", "Precision", "Recall"]

    first_legend = ax.legend(
        handles=colors_lines, labels=colors_labels, title="Class", loc="lower left"
    )
    ax.add_artist(first_legend)
    second_legend = ax.legend(handles=type_lines, labels=type_labels, title="Metric")
    ax.add_artist(second_legend)

    ax.axis("square")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Precision / Recall / F1-score")
    ax.set_title("Multiclass precision, recall and f1-score")
    ax.legend()
    fig.tight_layout()
    plt.show()

    return fig


def plot_roc_curve(results, figsize=(6, 6), palette="deep"):
    y_onehot_test = pd.get_dummies(results["labels"]).values
    y_score = results["probs"]

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette(palette)

    target_names = ["Gauche", "Centre", "Droite"]
    n_classes = 3

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

    colors = sns.color_palette(palette)
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
        )

    ax.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"Micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color=colors[3],
        linestyle=":",
        linewidth=4,
    )

    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"Macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color=colors[4],
        linestyle=":",
        linewidth=4,
    )

    ax.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    ax.axis("square")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    fig.tight_layout()
    plt.show()

    return fig


def plot_roc_curve_binary(results, figsize=(6, 6), palette="deep"):
    y_true = results["labels"]
    y_score = results["probs"][:, 1]

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette(palette)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ax.plot(
        fpr,
        tpr,
        label=f"ROC curve (AUC = {roc_auc:.2f})",
        color=colors[0],
        linestyle="-",
        linewidth=1.5,
    )

    ax.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    ax.axis("square")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    plt.show()

    return fig


def plot_precision_recall_curve_binary(results, figsize=(6, 6), palette="deep"):
    y_true = results["labels"]
    y_score = results["probs"][:, 1]

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette(palette)

    precision, recall, threshold = precision_recall_curve(y_true, y_score)
    f1 = 2 * (precision * recall) / (precision + recall)

    ax.plot([0] + list(threshold), f1, color=colors[2], linestyle="-", lw=1.5, label="F1-score")
    ax.plot(
        [0] + list(threshold), precision, color=colors[0], linestyle="-.", lw=1, label="Precision"
    )
    ax.plot([0] + list(threshold), recall, color=colors[1], linestyle="--", lw=1, label="Recall")

    ax.axis("square")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Precision / Recall / F1-score")
    ax.legend()
    fig.tight_layout()
    plt.show()

    return fig


def plot_confusion_matrix(results, figsize=(6, 6), normalized=None):
    y_pred = results["predictions"]
    y_true = results["labels"]

    fig, ax = plt.subplots(figsize=figsize)

    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        target_names = ["Gauche", "Droite"]
    else:
        target_names = ["Gauche", "Centre", "Droite"]

    cm = confusion_matrix(y_true, y_pred, normalize=normalized)

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        square=True,
        fmt=".2%" if normalized else ".0f",
        cbar=False,
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    plt.show()

    return fig


def plot_network_graph(net, device="cpu", model_name="model", path=None):
    save = path is not None
    if save:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        filename = path / "graph_architecture.png"

    input_data = net.example_input_array
    graph = draw_graph(
        net,
        input_data=input_data,
        graph_name=model_name,
        device=device,
        expand_nested=True,
        save_graph=save,
        filename=filename,
    )

    return graph.visual_graph
