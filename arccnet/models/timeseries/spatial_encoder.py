"""Spatial encoder for timeseries images using CNN backbone."""

import warnings

import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34


class SpatialEncoder(nn.Module):
    """
    CNN-based spatial encoder for multi-channel solar imagery.
    Uses ResNet34 backbone adapted for 10-channel input.

    Parameters
    ----------
    num_channels : int
        Number of input channels (default: 10 for 9 AIA + 1 HMI)
    pretrained : bool
        Whether to use ImageNet pretrained weights (default: True)
    feature_dim : int
        Output feature dimension (default: 512)
    freeze_backbone : bool
        Whether to freeze backbone weights (default: False)
    """

    def __init__(
        self,
        num_channels=10,
        pretrained=True,
        feature_dim=512,
        freeze_backbone=False,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.feature_dim = feature_dim

        # Load pretrained ResNet34
        if pretrained:
            try:
                weights = ResNet34_Weights.IMAGENET1K_V1
                backbone = resnet34(weights=weights)
            except Exception as exc:
                warnings.warn(
                    f"Could not load pretrained ResNet34 weights ({exc}). Falling back to random initialization."
                )
                backbone = resnet34(weights=None)
        else:
            backbone = resnet34(weights=None)

        # Adapt first conv layer for 10-channel input
        if num_channels != 3:
            old_conv = backbone.conv1
            # Create new conv with same params except input channels
            new_conv = nn.Conv2d(
                num_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

            # Initialize new conv weights
            if pretrained and num_channels >= 3:
                # Replicate RGB weights across channels
                with torch.no_grad():
                    # Get RGB weights
                    rgb_weights = old_conv.weight.data  # (64, 3, 7, 7)
                    # Repeat for extra channels
                    new_weights = rgb_weights.repeat(1, (num_channels + 2) // 3, 1, 1)
                    new_conv.weight.data = new_weights[:, :num_channels, :, :]
                    # Scale to maintain variance
                    new_conv.weight.data *= (3.0 / num_channels) ** 0.5
            else:
                # Random initialization
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

            backbone.conv1 = new_conv

        # Remove final FC layer
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection to feature_dim if needed
        backbone_dim = 512  # ResNet34 final layer
        if feature_dim != backbone_dim:
            self.proj = nn.Linear(backbone_dim, feature_dim)
        else:
            self.proj = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through spatial encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (B, feature_dim)
        """
        # Extract features
        features = self.backbone(x)  # (B, 512, H', W')

        # Global pooling
        pooled = self.global_pool(features)  # (B, 512, 1, 1)
        pooled = pooled.flatten(1)  # (B, 512)

        # Project to target dimension
        out = self.proj(pooled)  # (B, feature_dim)

        return out


def test_spatial_encoder():
    """Test spatial encoder with dummy input."""
    print("Testing SpatialEncoder...")

    # Create model
    model = SpatialEncoder(num_channels=10, pretrained=True, feature_dim=512)
    model.eval()

    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 10, 256, 512)

    # Forward pass
    with torch.no_grad():
        out = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: ({batch_size}, 512)")

    assert out.shape == (batch_size, 512), f"Shape mismatch: {out.shape}"
    print("✓ SpatialEncoder test passed!")

    return model


if __name__ == "__main__":
    test_spatial_encoder()
