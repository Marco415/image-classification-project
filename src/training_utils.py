import copy
import time

import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score
)


def get_logits(
    outputs
):

    if hasattr(
        outputs,
        "logits"
    ):

        return outputs.logits

    return outputs


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    accumulation_steps=1
):

    model.train()

    total_loss = 0.0

    targets = []
    predictions = []

    optimizer.zero_grad(
        set_to_none=True
    )

    for step, (
        images,
        labels
    ) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=device.type == "cuda"
        ):

            outputs = model(
                images
            )

            logits = get_logits(
                outputs
            )

            loss = criterion(
                logits,
                labels
            )

            loss_for_backward = (
                loss
                / accumulation_steps
            )

        if device.type == "cuda":

            scaler.scale(
                loss_for_backward
            ).backward()

        else:

            loss_for_backward.backward()

        if (
            (step + 1)
            % accumulation_steps
            == 0
        ):

            if device.type == "cuda":

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predicted = torch.argmax(
            logits,
            dim=1
        )

        targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        predictions.extend(
            predicted.detach()
            .cpu()
            .numpy()
        )

    # Handle final incomplete accumulation
    if (
        len(loader)
        % accumulation_steps
        != 0
    ):

        if device.type == "cuda":

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    return (
        average_loss,
        accuracy,
        macro_f1
    )


def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    targets = []
    predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=device.type == "cuda"
            ):

                outputs = model(
                    images
                )

                logits = get_logits(
                    outputs
                )

                loss = criterion(
                    logits,
                    labels
                )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predicted = torch.argmax(
                logits,
                dim=1
            )

            targets.extend(
                labels.cpu().numpy()
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    return (
        average_loss,
        accuracy,
        macro_f1
    )


def train_model(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    save_path,
    accumulation_steps=1
):

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    history = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
        "train_macro_f1": [],
        "validation_macro_f1": []
    }

    best_f1 = -1.0

    best_weights = copy.deepcopy(
        model.state_dict()
    )

    epochs_without_improvement = 0

    for epoch in range(
        epochs
    ):

        start_time = time.time()

        train_loss, train_accuracy, train_f1 = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                accumulation_steps
            )
        )

        (
            validation_loss,
            validation_accuracy,
            validation_f1
        ) = validate(
            model,
            validation_loader,
            criterion,
            device
        )

        if scheduler is not None:

            scheduler.step()

        history["train_loss"].append(
            train_loss
        )

        history["validation_loss"].append(
            validation_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["validation_accuracy"].append(
            validation_accuracy
        )

        history["train_macro_f1"].append(
            train_f1
        )

        history["validation_macro_f1"].append(
            validation_f1
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{epochs} | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Val Loss: "
            f"{validation_loss:.4f} | "
            f"Train Acc: "
            f"{train_accuracy:.4f} | "
            f"Val Acc: "
            f"{validation_accuracy:.4f} | "
            f"Val Macro F1: "
            f"{validation_f1:.4f} | "
            f"{elapsed:.1f}s"
        )

        if validation_f1 > best_f1:

            best_f1 = validation_f1

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "best_validation_macro_f1":
                        best_f1,

                    "epoch":
                        epoch + 1
                },
                save_path
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "\nEarly stopping."
            )

            break

    model.load_state_dict(
        best_weights
    )

    return model, history