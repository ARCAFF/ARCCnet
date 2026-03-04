"""Complete flare forecasting model integrating spatial and temporal components."""

import torch
import torch.nn as nn

from .config import NUM_CLASSES, TASK_TYPE, USE_TEMPORAL_TRANSFORMER
from .spatial_encoder import SpatialEncoder
from .temporal_transformer import TemporalTransformer


class FlareForecaster(nn.Module):
    """
    Complete model for solar flare forecasting from timeseries imagery.

    Architecture:
    1. Spatial encoder (CNN) extracts features from each timestep independently
    2. Temporal transformer processes the sequence of spatial features
    3. Task-specific head:
       - Multiclass: Predicts configured flare class labels (default: No-flare/C/M+) via CrossEntropyLoss
       - Regression: Predicts log flare counts via MSELoss

    Parameters
    ----------
    task_type : str
        'multiclass' or 'regression' (default: from config.TASK_TYPE)
    num_channels : int
        Number of input channels per image (default: 10)
    spatial_feature_dim : int
        Feature dimension from spatial encoder (default: 512)
    temporal_num_layers : int
        Number of transformer layers (default: 4)
    temporal_num_heads : int
        Number of attention heads (default: 8)
    temporal_dim_feedforward : int
        Feedforward dimension in transformer (default: 2048)
    temporal_dropout : float
        Dropout rate in transformer (default: 0.1)
    temporal_pooling : str
        Pooling strategy: 'cls', 'mean', 'last' (default: 'mean')
    output_dim : int
        Output dimension - num_classes for multiclass, 3 for regression
    pretrained_spatial : bool
        Whether to use pretrained spatial encoder (default: True)
    freeze_spatial : bool
        Whether to freeze spatial encoder (default: False)
    hidden_dims : list
        Hidden layer dimensions in prediction head (default: [256])
    dropout : float
        Dropout rate in prediction head (default: 0.3)
    """

    def __init__(
        self,
        task_type=None,
        num_channels=10,
        spatial_feature_dim=512,
        temporal_num_layers=4,
        temporal_num_heads=8,
        temporal_dim_feedforward=2048,
        temporal_dropout=0.1,
        temporal_pooling="mean",
        use_temporal_transformer=USE_TEMPORAL_TRANSFORMER,
        output_dim=None,
        pretrained_spatial=True,
        freeze_spatial=False,
        hidden_dims=[256],
        dropout=0.3,
    ):
        super().__init__()

        self.task_type = task_type or TASK_TYPE
        self.num_channels = num_channels
        self.spatial_feature_dim = spatial_feature_dim
        self.use_temporal_transformer = bool(use_temporal_transformer)
        if output_dim is None:
            output_dim = NUM_CLASSES if self.task_type == "multiclass" else 3
        self.output_dim = output_dim

        # Spatial encoder (CNN)
        self.spatial_encoder = SpatialEncoder(
            num_channels=num_channels,
            pretrained=pretrained_spatial,
            feature_dim=spatial_feature_dim,
            freeze_backbone=freeze_spatial,
        )

        # Temporal transformer (optional for PIT mode).
        self.temporal_transformer = None
        if self.use_temporal_transformer:
            self.temporal_transformer = TemporalTransformer(
                feature_dim=spatial_feature_dim,
                num_layers=temporal_num_layers,
                num_heads=temporal_num_heads,
                dim_feedforward=temporal_dim_feedforward,
                dropout=temporal_dropout,
                pooling=temporal_pooling,
            )

        # Task-specific prediction head
        layers = []
        in_dim = spatial_feature_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.prediction_head = nn.Sequential(*layers)

    @staticmethod
    def _select_last_valid_features(spatial_features, mask=None):
        """
        Select the last valid timestep feature for each sample.
        """
        if mask is None:
            return spatial_features[:, -1, :]

        lengths = mask.sum(dim=1).long().clamp(min=1)
        batch_idx = torch.arange(spatial_features.shape[0], device=spatial_features.device)
        return spatial_features[batch_idx, lengths - 1, :]

    def forward(self, x, mask=None):
        """
        Forward pass through complete model.

        Parameters
        ----------
        x : torch.Tensor
            Input timeseries of shape (B, T, C, H, W)
        mask : torch.Tensor, optional
            Valid timestep mask of shape (B, T)

        Returns
        -------
        torch.Tensor
            For multiclass: logits of shape (B, num_classes)
            For regression: predictions of shape (B, 3)
        """
        B, T, C, H, W = x.shape

        # Encode each timestep independently
        # Reshape to (B*T, C, H, W)
        x_flat = x.reshape(B * T, C, H, W)

        # Extract spatial features
        spatial_features = self.spatial_encoder(x_flat)  # (B*T, feature_dim)

        # Reshape back to (B, T, feature_dim)
        spatial_features = spatial_features.reshape(B, T, self.spatial_feature_dim)

        # Process sequence.
        if self.use_temporal_transformer:
            temporal_features = self.temporal_transformer(spatial_features, mask=mask)  # (B, feature_dim)
        else:
            temporal_features = self._select_last_valid_features(spatial_features, mask=mask)

        # Task-specific prediction
        output = self.prediction_head(temporal_features)  # (B, output_dim)

        return output

    def predict_proba(self, x, mask=None):
        """
        Get probability predictions (only for multiclass).

        Parameters
        ----------
        x : torch.Tensor
            Input timeseries of shape (B, T, C, H, W)
        mask : torch.Tensor, optional
            Valid timestep mask of shape (B, T)

        Returns
        -------
        torch.Tensor
            Class probabilities of shape (B, num_classes)
        """
        if self.task_type != "multiclass":
            raise ValueError("predict_proba only available for multiclass task")

        logits = self.forward(x, mask=mask)
        probs = torch.softmax(logits, dim=1)
        return probs


def test_flare_forecaster():
    """Test complete flare forecaster model."""
    print("Testing FlareForecaster...")

    # Create model
    model = FlareForecaster(
        num_channels=10,
        spatial_feature_dim=512,
        temporal_num_layers=4,
        temporal_num_heads=8,
        temporal_pooling="mean",
        output_dim=NUM_CLASSES,
        pretrained_spatial=True,
        hidden_dims=[256],
    )
    model.eval()

    # Create dummy input
    batch_size = 2
    timesteps = 6
    channels = 10
    height = 256
    width = 512

    x = torch.randn(batch_size, timesteps, channels, height, width)

    # Test forward pass
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        logits = model(x)
        probs = model.predict_proba(x)

    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")
    print(f"Probs range: [{probs.min():.3f}, {probs.max():.3f}]")

    assert logits.shape == (batch_size, NUM_CLASSES), f"Logits shape mismatch: {logits.shape}"
    assert probs.shape == (batch_size, NUM_CLASSES), f"Probs shape mismatch: {probs.shape}"
    assert (probs >= 0).all(), "Probabilities below 0"
    assert (probs <= 1).all(), "Probabilities above 1"

    # Test with mask
    mask = torch.ones(batch_size, timesteps, dtype=torch.bool)
    mask[0, 4:] = False  # Mask last 2 timesteps for first sample

    with torch.no_grad():
        logits_masked = model(x, mask=mask)

    print(f"Logits shape (masked): {logits_masked.shape}")
    assert logits_masked.shape == (batch_size, NUM_CLASSES), f"Logits shape mismatch: {logits_masked.shape}"

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\n✓ FlareForecaster test passed!")

    return model


if __name__ == "__main__":
    test_flare_forecaster()
