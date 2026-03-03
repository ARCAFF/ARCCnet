"""Temporal transformer for processing sequential features."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encoding to input.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, T, d_model)

        Returns
        -------
        torch.Tensor
            Output with positional encoding added (B, T, d_model)
        """
        return x + self.pe[:, : x.size(1), :]


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for processing temporal sequences.

    Parameters
    ----------
    feature_dim : int
        Dimension of input features (default: 512)
    num_layers : int
        Number of transformer encoder layers (default: 4)
    num_heads : int
        Number of attention heads (default: 8)
    dim_feedforward : int
        Dimension of feedforward network (default: 2048)
    dropout : float
        Dropout rate (default: 0.1)
    pooling : str
        Pooling strategy: 'cls', 'mean', 'last' (default: 'mean')
    """

    def __init__(
        self,
        feature_dim=512,
        num_layers=4,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.pooling = pooling

        # CLS token (learnable) if using cls pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Positional encoding
        self.pos_encoder = PositionalEncoding(feature_dim, max_len=100)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer norm
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x, mask=None):
        """
        Forward pass through temporal transformer.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (B, T, feature_dim)
        mask : torch.Tensor, optional
            Attention mask of shape (B, T) where True indicates valid positions

        Returns
        -------
        torch.Tensor
            Pooled temporal features of shape (B, feature_dim)
        """
        B, T, D = x.shape

        # Add CLS token if using cls pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, D)
            if mask is not None:
                # Add True for cls token
                cls_mask = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask for transformer
        # PyTorch transformer expects: True for positions to IGNORE
        if mask is not None:
            attn_mask = ~mask  # Invert: True -> ignore, False -> attend
        else:
            attn_mask = None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attn_mask)  # (B, T, D) or (B, T+1, D)

        # Pooling
        if self.pooling == "cls":
            out = x[:, 0, :]  # (B, D) - take CLS token
        elif self.pooling == "mean":
            if mask is not None:
                # Masked mean pooling
                mask_expanded = mask.unsqueeze(-1).float()  # (B, T, 1)
                sum_features = (x * mask_expanded).sum(dim=1)  # (B, D)
                count = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
                out = sum_features / count  # (B, D)
            else:
                out = x.mean(dim=1)  # (B, D)
        elif self.pooling == "last":
            if mask is not None:
                # Get last valid position for each sample
                lengths = mask.sum(dim=1).long()  # (B,)
                batch_idx = torch.arange(B, device=x.device)
                out = x[batch_idx, lengths - 1, :]  # (B, D)
            else:
                out = x[:, -1, :]  # (B, D) - take last timestep
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Final norm
        out = self.norm(out)

        return out


def test_temporal_transformer():
    """Test temporal transformer with dummy input."""
    print("Testing TemporalTransformer...")

    # Create model
    model = TemporalTransformer(
        feature_dim=512,
        num_layers=4,
        num_heads=8,
        pooling="mean",
    )
    model.eval()

    # Create dummy input
    batch_size = 2
    timesteps = 6
    x = torch.randn(batch_size, timesteps, 512)

    # Test without mask
    with torch.no_grad():
        out = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape (no mask): {out.shape}")
    assert out.shape == (batch_size, 512), f"Shape mismatch: {out.shape}"

    # Test with mask
    mask = torch.ones(batch_size, timesteps, dtype=torch.bool)
    mask[0, 4:] = False  # Mask last 2 timesteps for first sample

    with torch.no_grad():
        out_masked = model(x, mask=mask)

    print(f"Output shape (with mask): {out_masked.shape}")
    assert out_masked.shape == (batch_size, 512), f"Shape mismatch: {out_masked.shape}"

    # Test CLS pooling
    model_cls = TemporalTransformer(
        feature_dim=512,
        num_layers=2,
        num_heads=8,
        pooling="cls",
    )
    model_cls.eval()

    with torch.no_grad():
        out_cls = model_cls(x)

    print(f"Output shape (CLS pooling): {out_cls.shape}")
    assert out_cls.shape == (batch_size, 512), f"Shape mismatch: {out_cls.shape}"

    print("✓ TemporalTransformer test passed!")

    return model


if __name__ == "__main__":
    test_temporal_transformer()
