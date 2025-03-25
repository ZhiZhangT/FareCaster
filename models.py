import torch.nn as nn

# GIVEN 176,000 samples, 5 input features, 1 output feature
# technically, number of neurons should be in the range of (1465, 14665)


# GRU-based model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Regression output

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # Use the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# LSTM-based model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# -------------------------------
# GRU-based deep model
# -------------------------------
class GRUModelDeep(nn.Module):
    def __init__(
        self, input_size, hidden_size=32, num_layers=2, fc_hidden_sizes=[64, 32]
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
        fc_layers.append(nn.Linear(prev_size, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # Use the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# -------------------------------
# LSTM-based deep model
# -------------------------------
class LSTMModelDeep(nn.Module):
    def __init__(
        self, input_size, hidden_size=32, num_layers=2, fc_hidden_sizes=[64, 32]
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
        fc_layers.append(nn.Linear(prev_size, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Use the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out
