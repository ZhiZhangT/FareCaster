import torch.nn as nn
import math
import torch

# GIVEN 149,568 samples, 14 input features, 1 output feature
# technically, number of neurons should be in the range of (499, 4985)


# GRU-based model
class GRUModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # Use the output of the last time step
        out = out[:, -1, :]  # shape: (batch, hidden_size)
        out = self.fc(out)  # shape: (batch, output_size)
        out = out.unsqueeze(1)  # shape: (batch, 1, output_size)
        return out


# LSTM-based model
class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # shape: (batch, hidden_size)
        out = self.fc(out)  # shape: (batch, output_size)
        out = out.unsqueeze(1)  # shape: (batch, 1, output_size)
        return out


# -------------------------------
# GRU-based deep model
# -------------------------------
class GRUModelDeep(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=32,
        num_layers=2,
        fc_hidden_sizes=[64, 32],
    ):
        """
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of stacked GRU layers.
            fc_hidden_sizes (list): List of hidden layer sizes for the fully connected (FC) block.
        """
        super(GRUModelDeep, self).__init__()
        # Initialize GRU with potential dropout if num_layers > 1
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )

        # Build a multi-layer fully connected network
        fc_layers = []
        prev_size = hidden_size
        for hidden in fc_hidden_sizes:
            fc_layers.append(nn.Linear(prev_size, hidden))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.2))
            prev_size = hidden
        # Final output layer for regression
        fc_layers.append(nn.Linear(prev_size, output_size))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # Use the output of the last time step
        out = out[:, -1, :]  # shape: (batch, hidden_size)
        out = self.fc(out)  # shape: (batch, output_size)
        out = out.unsqueeze(1)  # shape: (batch, 1, output_size)
        return out


# -------------------------------
# LSTM-based deep model
# -------------------------------
class LSTMModelDeep(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=32,
        num_layers=2,
        fc_hidden_sizes=[64, 32],
    ):
        """
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of stacked LSTM layers.
            fc_hidden_sizes (list): List of hidden layer sizes for the fully connected (FC) block.
        """
        super(LSTMModelDeep, self).__init__()
        # Initialize LSTM with dropout if num_layers > 1
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )

        # Build a multi-layer fully connected network
        fc_layers = []
        prev_size = hidden_size
        for hidden in fc_hidden_sizes:
            fc_layers.append(nn.Linear(prev_size, hidden))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.2))
            prev_size = hidden
        # Final output layer for regression
        fc_layers.append(nn.Linear(prev_size, output_size))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Use the output of the last time step
        out = out[:, -1, :]  # shape: (batch, hidden_size)
        out = self.fc(out)  # shape: (batch, output_size)
        out = out.unsqueeze(1)  # shape: (batch, 1, output_size)
        return out


# -------------------------------
# LSTM Encoder-Decoder model
# -------------------------------
class LSTMEncoderDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=32,
        num_layers=2,
        dropout=0.2,
    ):
        """
        LSTM-based Encoder-Decoder model for time series forecasting.
        
        Args:
            input_size (int): Number of input features
            output_size (int): Number of output features to predict
            hidden_size (int): Size of hidden layers
            num_layers (int): Number of LSTM layers in encoder and decoder
            dropout (float): Dropout probability (applied if num_layers > 1)
        """
        super(LSTMEncoderDecoder, self).__init__()
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=1,  # Decoder input is a single value
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass through the encoder-decoder model.
        
        Args:
            x: Input tensor with shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor with shape (batch_size, 1, output_size)
        """
        batch_size = x.size(0)
        
        # Encode the input sequence
        _, (hidden, cell) = self.encoder(x)
        
        # Initial decoder input (zeros)
        decoder_input = torch.zeros(batch_size, 1, 1, device=x.device)
        
        # Pass through decoder with encoder's hidden state
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        
        # Project to output size
        output = self.fc(decoder_output)  # Shape: [batch_size, 1, output_size]
        
        return output


# -------------------------------
# Transformer model
# -------------------------------
class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        num_layers=2,
        nhead=4,
        dropout=0.1,
    ):
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
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

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

        # Reshape to (batch_size, 1, output_size)
        x = x.unsqueeze(1)

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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer to be saved with model
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (batch_size, sequence_length, d_model)

        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
