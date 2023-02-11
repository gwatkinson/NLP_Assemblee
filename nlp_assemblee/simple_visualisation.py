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

    colors = sns.color_palette("deep")
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
    ax.legend()
    plt.show()

    return fig
