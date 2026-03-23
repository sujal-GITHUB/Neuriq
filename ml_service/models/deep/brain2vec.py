"""
NeuroAnxiety — Brain2Vec Architecture
======================================
CNN + LSTM + Self-Attention hybrid from arXiv:2506.11179 (Mynoddin et al., 2025).

Architecture:
  Input: (batch, channels, timepoints, 1)
  → 3 × Conv2D blocks: [32, 64, 128] filters, (1,5) kernel, BN, ReLU, MaxPool
  → Reshape: (batch, 128, reduced_time)
  → LSTM (hidden=256, layers=1)
  → Self-Attention (Q/K/V, scaled dot-product)
  → Global Average Pooling
  → FC(128) → ReLU → Dropout(0.4) → FC(n_classes) → Softmax

This is the PRIMARY deep learning model of the system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """Scaled dot-product self-attention layer."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            attended: (batch, seq_len, hidden_dim)
        """
        Q = self.query(x)  # (batch, seq, hidden)
        K = self.key(x)
        V = self.value(x)
        
        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch, seq, seq)
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.bmm(attention_weights, V)  # (batch, seq, hidden)
        
        return attended


class Brain2Vec(nn.Module):
    """
    Brain2Vec: CNN + LSTM + Self-Attention for EEG anxiety classification.
    
    From: Mynoddin et al. (2025), arXiv:2506.11179
    """
    
    def __init__(
        self,
        n_channels: int = 32,
        n_timepoints: int = 256,
        n_classes: int = 3,
        conv_filters: list = None,
        lstm_hidden: int = 256,
        dropout: float = 0.4
    ):
        super().__init__()
        
        if conv_filters is None:
            conv_filters = [32, 64, 128]
        
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes
        
        # Conv2D Block 1
        self.conv1 = nn.Conv2d(1, conv_filters[0], kernel_size=(1, 5), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(conv_filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        
        # Conv2D Block 2
        self.conv2 = nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=(1, 5), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(conv_filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        
        # Conv2D Block 3
        self.conv3 = nn.Conv2d(conv_filters[1], conv_filters[2], kernel_size=(1, 5), padding=(0, 2))
        self.bn3 = nn.BatchNorm2d(conv_filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        
        # Compute reduced time dimension after pooling
        reduced_time = n_timepoints // 8  # 3 MaxPool(1,2) layers
        
        # LSTM
        lstm_input_size = conv_filters[2] * n_channels
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        
        # Self-Attention
        self.attention = SelfAttention(lstm_hidden)
        
        # Classification head
        self.fc1 = nn.Linear(lstm_hidden, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, timepoints)
               or (batch, 1, channels, timepoints)
        
        Returns:
            logits: (batch, n_classes)
        """
        # Ensure 4D input: (batch, 1, channels, timepoints)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Conv blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))   # (B, 32, C, T/2)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))   # (B, 64, C, T/4)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))   # (B, 128, C, T/8)
        
        # Reshape for LSTM: (batch, time_steps, features)
        batch_size = x.size(0)
        # x shape: (B, 128, C, T/8)
        x = x.permute(0, 3, 1, 2)  # (B, T/8, 128, C)
        x = x.reshape(batch_size, x.size(1), -1)  # (B, T/8, 128*C)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T/8, lstm_hidden)
        
        # Self-Attention
        attended = self.attention(lstm_out)  # (B, T/8, lstm_hidden)
        
        # Global Average Pooling
        pooled = torch.mean(attended, dim=1)  # (B, lstm_hidden)
        
        # Classification
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


def create_brain2vec(
    n_channels: int = 32,
    n_timepoints: int = 512,
    n_classes: int = 3
) -> Brain2Vec:
    """Factory function to create Brain2Vec model."""
    from config import B2V_CONV_FILTERS, B2V_LSTM_HIDDEN, B2V_DROPOUT
    
    return Brain2Vec(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=n_classes,
        conv_filters=B2V_CONV_FILTERS,
        lstm_hidden=B2V_LSTM_HIDDEN,
        dropout=B2V_DROPOUT
    )
