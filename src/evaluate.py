import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_fscore_support,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import config


def evaluate_model(model, validation_generator):
    val_steps = validation_generator.samples // validation_generator.batch_size
    predictions = model.predict(validation_generator, steps=val_steps + 1, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = validation_generator.classes

    class_labels = list(validation_generator.class_indices.keys())
    num_classes = len(class_labels)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"Validation Accuracy:  {acc*100:.2f}%")
    print(f"Weighted Precision:  {precision:.4f}")
    print(f"Weighted Recall:     {recall:.4f}")
    print(f"Weighted F1-Score:   {f1:.4f}")
    print(f"Macro Precision:     {precision_macro:.4f}")
    print(f"Macro Recall:        {recall_macro:.4f}")
    print(f"Macro F1-Score:      {f1_macro:.4f}")
    print(f"{'='*50}\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_true": y_true,
        "predictions": predictions,
        "class_labels": class_labels,
    }


def plot_confusion_matrix(cm, class_labels, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("StyleSense Classification Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_auc(y_true, predictions, class_labels, save_path=None):
    n_classes = len(class_labels)
    y_true_bin = tf.keras.utils.to_categorical(y_true, num_classes=n_classes)

    fpr = {}
    tpr = {}
    roc_auc = {}

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label=f"{class_labels[i]} (AUC = {roc_auc[i]:.2f})",
        )

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        lw=2,
        linestyle="--",
        label=f"Micro-average (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
    )

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("StyleSense ROC & AUC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"ROC curve saved to {save_path}")
    plt.close()

    return roc_auc


def plot_training_history(history, save_path_prefix=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    ax1.set_title("StyleSense Accuracy Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    ax2.set_title("StyleSense Loss Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_training_history.png", dpi=150)
        print(f"Training history saved to {save_path_prefix}_training_history.png")
    plt.close()


def plot_prediction_confidence(predictions, y_true, class_labels, save_path=None):
    plt.figure(figsize=(10, 6))
    max_probs = np.max(predictions, axis=1)
    plt.hist(max_probs, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
    plt.axvline(
        x=np.mean(max_probs),
        color="red",
        linestyle="--",
        label=f"Mean Confidence: {np.mean(max_probs):.3f}",
    )
    plt.title("StyleSense Prediction Confidence Distribution")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confidence distribution saved to {save_path}")
    plt.close()
