"""
NeuroAnxiety — EEGNet Architecture
===================================
Compact CNN for EEG signals from Lawhern et al. (2018).

Architecture:
  Input: (batch, 1, channels, timepoints)
  Block 1:
    → Conv2D(F1=8, (1,64), 'same') → BN
    → DepthwiseConv2D(F1, (channels,1), D=2) → BN → ELU → AvgPool → Dropout(0.5)
  Block 2:
    → SeparableConv2D(F2=16, (1,16)) → BN → ELU → AvgPool → Dropout(0.5)
  → Flatten → Dense(n_classes)

Implemented in PyTorch following the original paper specification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGNet(nn.Module):
    """
    EEGNet: Compact CNN for EEG (Lawhern et al., 2018).
    
    F1: number of temporal filters (default 8)
    F2: number of pointwise filters (default 16)
    D: depth multiplier for depthwise conv (default 2)
    """
    
    def __init__(
        self,
        n_channels: int = 32,
        n_timepoints: int = 512,
        n_classes: int = 3,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes
        
        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise Conv
        self.depthwise = nn.Conv2d(
            F1, F1 * D, (n_channels, 1),
            groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        
        # Block 2
        self.separable = SeparableConv2d(F1 * D, F2, (1, 16), padding=(0, 8))
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        
        # Compute flattened size
        # After conv1: (B, F1, C, T) → After depthwise: (B, F1*D, 1, T)
        # After pool1: (B, F1*D, 1, T/4)
        # After separable: (B, F2, 1, T/4)
        # After pool2: (B, F2, 1, T/32)
        flatten_size = F2 * (n_timepoints // 32)
        
        # Classification
        self.classifier = nn.Linear(flatten_size, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, timepoints) or (batch, 1, channels, timepoints)
        Returns:
            logits: (batch, n_classes)
        """
        # Ensure 4D: (batch, 1, channels, timepoints)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Block 1
        x = self.conv1(x)                    # (B, F1, C, T)
        x = self.bn1(x)
        x = self.depthwise(x)                # (B, F1*D, 1, T)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)                    # (B, F1*D, 1, T/4)
        x = self.drop1(x)
        
        # Block 2
        x = self.separable(x)                # (B, F2, 1, T/4)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)                    # (B, F2, 1, T/32)
        x = self.drop2(x)
        
        # Flatten and classify
        x = x.flatten(1)
        logits = self.classifier(x)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


def create_eegnet(
    n_channels: int = 32,
    n_timepoints: int = 512,
    n_classes: int = 3
) -> EEGNet:
    """Factory function."""
    from config import EEGNET_F1, EEGNET_F2, EEGNET_D, EEGNET_DROPOUT
    
    return EEGNet(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=n_classes,
        F1=EEGNET_F1,
        F2=EEGNET_F2,
        D=EEGNET_D,
        dropout=EEGNET_DROPOUT
    )
