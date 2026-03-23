"""
NeuroAnxiety — CNN-BiLSTM with Attention
=========================================
Convolutional + Bidirectional LSTM with attention mechanism.

Architecture:
  Input: (batch, channels, timepoints)
  → Conv1D(64, k=3) → ReLU → BN → MaxPool
  → Conv1D(128, k=3) → ReLU → BN → MaxPool
  → Reshape for LSTM
  → BiLSTM(hidden=256, layers=2, dropout=0.3)
  → Attention (learned weights over LSTM output)
  → FC(128) → ReLU → Dropout(0.5) → FC(n_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Additive attention mechanism over sequence."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        Returns:
            weighted_sum: (batch, hidden_dim)
        """
        weights = self.attention(lstm_output)        # (B, seq, 1)
        weights = F.softmax(weights, dim=1)          # (B, seq, 1)
        weighted = lstm_output * weights              # (B, seq, hidden)
        return torch.sum(weighted, dim=1)             # (B, hidden)


class CNNLSTM(nn.Module):
    """CNN-BiLSTM with Attention for EEG classification."""
    
    def __init__(
        self,
        n_channels: int = 32,
        n_timepoints: int = 512,
        n_classes: int = 3,
        conv1_filters: int = 64,
        conv2_filters: int = 128,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        fc_dropout: float = 0.5
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Conv1D blocks (applied per channel, then combined)
        self.conv1 = nn.Conv1d(n_channels, conv1_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv2_filters)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Reduced time after 2 MaxPool
        reduced_time = n_timepoints // 4
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=conv2_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0
        )
        
        # Attention
        self.attention = Attention(lstm_hidden * 2)  # *2 for bidirectional
        
        # Classification head
        self.fc1 = nn.Linear(lstm_hidden * 2, 128)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(128, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, timepoints)
        Returns:
            logits: (batch, n_classes)
        """
        # Conv blocks: (B, C, T) → (B, filters, T/4)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Reshape for LSTM: (B, filters, T') → (B, T', filters)
        x = x.permute(0, 2, 1)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (B, T', 2*hidden)
        
        # Attention
        context = self.attention(lstm_out)  # (B, 2*hidden)
        
        # Classification
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


def create_cnn_lstm(
    n_channels: int = 32,
    n_timepoints: int = 512,
    n_classes: int = 3
) -> CNNLSTM:
    """Factory function."""
    from config import CL_CONV1_FILTERS, CL_CONV2_FILTERS, CL_LSTM_HIDDEN, CL_LSTM_LAYERS, CL_LSTM_DROPOUT
    
    return CNNLSTM(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=n_classes,
        conv1_filters=CL_CONV1_FILTERS,
        conv2_filters=CL_CONV2_FILTERS,
        lstm_hidden=CL_LSTM_HIDDEN,
        lstm_layers=CL_LSTM_LAYERS,
        lstm_dropout=CL_LSTM_DROPOUT
    )
