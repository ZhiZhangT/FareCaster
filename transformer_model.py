import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, nhead=4, dropout=0.1):
        """
        TransformerModel for time series prediction.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden dimension in transformer
            num_layers: Number of transformer encoder layers
            nhead: Number of heads in multi-head attention
            dropout: Dropout probability
        """
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Position encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor with shape (batch_size, 1)
        """
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Use the output from the last time step for prediction
        x = x[:, -1, :]
        
        # Project to output dimension
        x = self.output_layer(x)
        
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Positional encoding for transformer model.
        
        Args:
            d_model: Hidden dimension size
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer to be saved with model
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (batch_size, sequence_length, d_model)
        
        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)