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

class CustomCNN(nn.Module):
    """
    Small VGG-style CNN trained from scratch (no pretrained weights).
    4 blocks of [Conv-BN-ReLU x2 -> MaxPool], global average pooling, dropout, linear.
    Input: [B, 13, 64, 64]  ->  Output: [B, num_classes]
    """

    def __init__(self, num_classes=NUM_CLASSES, input_channels=NUM_BANDS, dropout=0.3):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(input_channels, 32),   # 64 -> 32
            block(32, 64),               # 32 -> 16
            block(64, 128),              # 16 -> 8
            block(128, 256),             # 8  -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def create_custom_cnn(num_classes=NUM_CLASSES, input_channels=NUM_BANDS):
    return CustomCNN(num_classes=num_classes, input_channels=input_channels)