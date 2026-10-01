import time

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)


def predict(
    model,
    loader,
    device
):

    model.eval()

    all_targets = []
    all_predictions = []

    inference_start = time.perf_counter()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            if hasattr(
                outputs,
                "logits"
            ):

                outputs = outputs.logits

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_targets.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    return (
        np.array(all_targets),
        np.array(all_predictions),
        inference_time
    )


def evaluate_predictions(
    targets,
    predictions,
    class_names
):

    report = classification_report(
        targets,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    results = {

        "accuracy":
            accuracy_score(
                targets,
                predictions
            ),

        "macro_precision":
            precision_score(
                targets,
                predictions,
                average="macro",
                zero_division=0
            ),

        "macro_recall":
            recall_score(
                targets,
                predictions,
                average="macro",
                zero_division=0
            ),

        "macro_f1":
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0
            ),

        "weighted_f1":
            f1_score(
                targets,
                predictions,
                average="weighted",
                zero_division=0
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                targets,
                predictions
            )
    }

    return (
        results,
        report
    )


def get_confusion_matrix(
    targets,
    predictions
):

    return confusion_matrix(
        targets,
        predictions
    )