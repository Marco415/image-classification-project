import torch
import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)


NUM_BANDS = 13
NUM_CLASSES = 10


def initialise_multispectral_conv(
    original_weight,
    new_weight
):
    """
    Initialise a 13-channel convolution using
    pretrained RGB weights.

    First 3 channels:
        copied directly from RGB pretrained weights.

    Remaining channels:
        initialized using the mean RGB filter.
    """

    with torch.no_grad():

        new_weight[
            :,
            :3,
            :,
            :
        ].copy_(
            original_weight
        )

        if new_weight.shape[1] > 3:

            mean_weight = (
                original_weight.mean(
                    dim=1,
                    keepdim=True
                )
            )

            extra_channels = (
                new_weight.shape[1] - 3
            )

            new_weight[
                :,
                3:,
                :,
                :
            ].copy_(
                mean_weight.repeat(
                    1,
                    extra_channels,
                    1,
                    1
                )
            )


def create_resnet50(
    num_classes=NUM_CLASSES,
    input_channels=NUM_BANDS
):

    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    old_conv = model.conv1

    new_conv = nn.Conv2d(
        input_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    initialise_multispectral_conv(
        old_conv.weight,
        new_conv.weight
    )

    model.conv1 = new_conv

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


def create_efficientnetv2(
    num_classes=NUM_CLASSES,
    input_channels=NUM_BANDS
):

    model = efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.DEFAULT
    )

    old_conv = model.features[0][0]

    new_conv = nn.Conv2d(
        input_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    initialise_multispectral_conv(
        old_conv.weight,
        new_conv.weight
    )

    model.features[0][0] = new_conv

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model


def create_vit(
    num_classes=NUM_CLASSES,
    input_channels=NUM_BANDS
):

    model = vit_b_16(
        weights=ViT_B_16_Weights.DEFAULT
    )

    old_projection = (
        model.conv_proj
    )

    new_projection = nn.Conv2d(
        input_channels,
        old_projection.out_channels,
        kernel_size=old_projection.kernel_size,
        stride=old_projection.stride,
        padding=old_projection.padding,
        bias=False
    )

    initialise_multispectral_conv(
        old_projection.weight,
        new_projection.weight
    )

    model.conv_proj = new_projection

    model.heads.head = nn.Linear(
        model.heads.head.in_features,
        num_classes
    )

    return model