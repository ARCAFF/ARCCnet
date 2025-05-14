import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
)


class HierarchicalResNet(nn.Module):
    def __init__(self, num_classes_Z: int, num_classes_P: int, num_classes_C: int, resnet_version: str = "resnet18"):
        """
        ResNet model for multi-level classification with optional Teacher Forcing.

        Args:
            num_classes_Z (int): Number of classes for the Z component.
            num_classes_P (int): Number of classes for the P component.
            num_classes_C (int): Number of classes for the C component.
            resnet_version (str): Which ResNet variant to use (e.g., 'resnet18', 'resnet34', 'resnet50', etc.).
        """
        super().__init__()

        # Mapping of ResNet version strings to corresponding model functions and weights
        resnet_versions = {
            "resnet18": (models.resnet18, ResNet18_Weights.DEFAULT, 512),
            "resnet34": (models.resnet34, ResNet34_Weights.DEFAULT, 512),
            "resnet50": (models.resnet50, ResNet50_Weights.DEFAULT, 2048),
            "resnet101": (models.resnet101, ResNet101_Weights.DEFAULT, 2048),
            "resnet152": (models.resnet152, ResNet152_Weights.DEFAULT, 2048),
            "wide_resnet50_2": (models.wide_resnet50_2, Wide_ResNet50_2_Weights.DEFAULT, 2048),
            "wide_resnet101_2": (models.wide_resnet101_2, Wide_ResNet101_2_Weights.DEFAULT, 2048),
            "resnext50_32x4d": (models.resnext50_32x4d, ResNeXt50_32X4D_Weights.DEFAULT, 2048),
            "resnext101_32x8d": (models.resnext101_32x8d, ResNeXt101_32X8D_Weights.DEFAULT, 2048),
        }

        if resnet_version not in resnet_versions:
            raise ValueError(
                f"Unsupported resnet_version: {resnet_version}. Supported versions are: {list(resnet_versions.keys())}"
            )

        resnet_fn, resnet_weights, backbone_output_size = resnet_versions[resnet_version]
        self.resnet = resnet_fn(weights=resnet_weights)

        # Modify the first convolutional layer to accept single-channel (monochrome) images
        self.resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the original fully connected layer
        self.resnet.fc = nn.Identity()

        # Additional fully connected layer to project ResNet features into a lower dimension
        self.fc_features = nn.Linear(backbone_output_size, 224)

        # Hierarchical classification heads
        #    Z -> P -> C
        self.fc_Z = nn.Linear(224, num_classes_Z)
        self.fc_P = nn.Linear(224 + num_classes_Z, num_classes_P)
        self.fc_C = nn.Linear(224 + num_classes_Z + num_classes_P, num_classes_C)

    def forward(
        self,
        x: torch.Tensor,
        Z_true: torch.Tensor = None,
        P_true: torch.Tensor = None,
        teacher_forcing_ratio=None,
    ) -> tuple:
        """
        Forward pass of the UnifiedResNet model with optional Teacher Forcing.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, H, W).
            Z_true (torch.Tensor, optional): Ground truth labels for Z component. Shape: (B,).
            P_true (torch.Tensor, optional): Ground truth labels for P component. Shape: (B,).
            teacher_forcing_ratio (float, optional): Probability of using ground truth labels. If None, Teacher
                                                     Forcing is disabled.

        Returns:
            tuple: Contains probabilities for Z, P, and C classification heads.
                   (Z_probs, P_probs, C_probs)
        """
        # Pass through ResNet backbone
        features = self.resnet(x)  # shape: (B, 512)

        # Project features to 224 dimensions with ReLU activation
        features_224 = F.relu(self.fc_features(features))  # shape: (B, 224)

        # Z-component prediction
        Z_logits = self.fc_Z(features_224)  # shape: (B, num_classes_Z)
        Z_probs = F.softmax(Z_logits, dim=1)  # shape: (B, num_classes_Z)

        # Determine if teacher forcing is enabled for Z
        if self.training and teacher_forcing_ratio is not None and Z_true is not None:
            use_teacher_forcing_Z = random.random() < teacher_forcing_ratio
            Z_input = (
                F.one_hot(Z_true, num_classes=Z_probs.size(1)).float() if use_teacher_forcing_Z else Z_probs
            )  # shape: (B, num_classes_Z)
        else:
            Z_input = Z_probs

        # P-component prediction (depends on Z)
        P_input = torch.cat([features_224, Z_input], dim=1)  # shape: (B, 224 + num_classes_Z)
        P_logits = self.fc_P(P_input)  # shape: (B, num_classes_P)
        P_probs = F.softmax(P_logits, dim=1)  # shape: (B, num_classes_P)

        # Determine if teacher forcing is enabled for P
        if self.training and teacher_forcing_ratio is not None and P_true is not None:
            use_teacher_forcing_P = random.random() < teacher_forcing_ratio
            P_input_final = (
                F.one_hot(P_true, num_classes=P_probs.size(1)).float() if use_teacher_forcing_P else P_probs
            )  # shape: (B, num_classes_P)
        else:
            P_input_final = P_probs

        # C-component prediction (depends on Z and P)
        C_input = torch.cat(
            [features_224, Z_input, P_input_final], dim=1
        )  # shape: (B, 224 + num_classes_Z + num_classes_P)
        C_logits = self.fc_C(C_input)  # shape: (B, num_classes_C)
        C_probs = F.softmax(C_logits, dim=1)  # shape: (B, num_classes_C)

        return Z_probs, P_probs, C_probs
