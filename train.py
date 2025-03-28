import math
import torch
import torch.nn as nn
import time
import os
import sys
from datetime import datetime
from models import GRUModel, LSTMModel, GRUModelDeep, LSTMModelDeep
from torch.utils.data import TensorDataset, DataLoader
from preprocess import get_data
import matplotlib.pyplot as plt
import json
import numpy as np


# Create a timestamped run directory
def create_run_directory(run_name="model_run"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"saved/{run_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# Custom logger to save output to file and print to console
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        
def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


# Modified plot_losses function to save in the run directory
def plot_losses(losses, save_dir, model_type):
    train_losses = losses["train_losses"]
    val_losses = losses["val_losses"]

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_type} Training and Validation Loss")
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"{model_type}_loss_plot.png"))
    plt.close()

    # Save the losses to a JSON file
    with open(os.path.join(save_dir, f"{model_type}_losses.json"), "w") as f:
        json.dump(losses, f)


# Modified training function to accept save directory
def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    lr=0.001,
    save_dir=None,
    model_type="model",
    loss_criteria=nn.MSELoss(),
):
    start_time = time.time()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")
    model.to(device)

    criterion = loss_criteria
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_epoch = -1

    # Lists to track losses
    train_losses = []
    val_losses = []

    model_save_path = (
        os.path.join(save_dir, f"best_{model_type}_model.pt") if save_dir else None
    )

    for epoch in range(num_epochs):
        start_time_epoch = time.time()
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Save the model if validation loss is the best seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            if model_save_path is not None:
                torch.save(model.state_dict(), model_save_path)
                print(
                    f"Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}"
                )

        print(f"Time taken for epoch {epoch+1}: {time.time() - start_time_epoch:.2f}s")

    # Optionally, load the best model state at the end of training
    if best_epoch != -1 and model_save_path is not None:
        model.load_state_dict(torch.load(model_save_path))
        print(
            f"Loaded best model from epoch {best_epoch+1} with val loss {best_val_loss:.4f}"
        )

    print(f"Total time taken: {time.time() - start_time:.2f}s")

    # Return the model and the loss history
    return model, {"train_losses": train_losses, "val_losses": val_losses}


def evaluate_model(model, test_loader, loss_criteria):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Evaluating on device: {device}")
    model.to(device)
    model.eval()
    criterion = loss_criteria
    test_loss = 0.0
    total_abs_error = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            total_abs_error += torch.sum(torch.abs(predictions - y_batch)).item()
    test_loss /= len(test_loader.dataset)
    mae = total_abs_error / len(test_loader.dataset)
    return test_loss, mae



def create_sliding_windows(data, sequence_length):
    """
    Create sliding windows of specified length from input data.
    
    Args:
        data: Input data of shape (batch_size, seq_len, feature_dim)
        sequence_length: Length of the subsequences to create
        
    Returns:
        Array of shape (new_batch_size, sequence_length, feature_dim)
    """
    batch_size, seq_len, feature_dim = data.shape
    if sequence_length >= seq_len:
        return data  # No sliding needed if requested length is equal or longer
    
    # Number of sliding windows per original sequence
    num_windows = seq_len - sequence_length + 1
    
    # Initialize output array
    windowed_data = []
    
    # Create sliding windows
    for i in range(batch_size):
        for j in range(num_windows):
            window = data[i, j:j+sequence_length, :]
            windowed_data.append(window)
    
    # Stack along batch dimension
    return np.stack(windowed_data, axis=0)

# -------------------------------
# Main execution flow with run directory
# -------------------------------

def main(
    batch_size=32,
    num_epochs=50,
    lr=0.005,
    num_layers=5,
    fc_hidden_sizes=[512, 256, 128, 64, 32],
    sequence_length=30  # Default to full length
):
    # Create run directory
    run_name = "price_prediction"
    run_dir = create_run_directory(run_name)

    # Set up logging to capture all print statements
    sys.stdout = Logger(os.path.join(run_dir, "training_log.txt"))

    print(f"Starting training run in directory: {run_dir}")
    
    # Log hyperparameters
    print(f"Hyperparameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_epochs: {num_epochs}")
    print(f"  learning rate: {lr}")
    print(f"  num_layers: {num_layers}")
    print(f"  fc_hidden_sizes: {fc_hidden_sizes}")
    print(f"  sequence_length: {sequence_length}")

    # Load and preprocess data
    data = get_data(sequence_length=sequence_length)

    X_train_scaled = data["X_train_scaled"]
    y_train = data["y_train"]
    X_val_scaled = data["X_val_scaled"]
    y_val = data["y_val"]
    X_test_scaled = data["X_test_scaled"]
    y_test = data["y_test"]
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_size = X_train_scaled.shape[2]  # number of features
    print(f"input_size (number of features): {input_size}")

    # Train GRU Model
    print("Training GRU model...")
    gru_model = GRUModelDeep(
        input_size=input_size, num_layers=num_layers, fc_hidden_sizes=fc_hidden_sizes
    )
    gru_model, gru_losses = train_model(
        gru_model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=lr,
        save_dir=run_dir,
        model_type="gru",
        loss_criteria=nn.MSELoss(),
    )

    plot_losses(gru_losses, run_dir, "gru")

    best_gru_val_loss, best_gru_val_mae = evaluate_model(gru_model, val_loader, nn.MSELoss())
    print(
        f"BEST GRU Model Val Loss: {best_gru_val_loss:.4f}, Val MAE: {best_gru_val_mae:.4f}"
    )

    # Train LSTM Model
    print("\nTraining LSTM model...")
    lstm_model = LSTMModelDeep(
        input_size=input_size, num_layers=num_layers, fc_hidden_sizes=fc_hidden_sizes
    )
    lstm_model, lstm_losses = train_model(
        lstm_model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=lr,
        save_dir=run_dir,
        model_type="lstm",
        loss_criteria=nn.MSELoss(),
    )

    plot_losses(lstm_losses, run_dir, "lstm")

    best_lstm_val_loss, best_lstm_val_mae = evaluate_model(lstm_model, val_loader, nn.MSELoss())
    print(
        f"BEST LSTM Model Val Loss: {best_lstm_val_loss:.4f}, Val MAE: {best_lstm_val_mae:.4f}"
    )

    # Save experiment summary
    summary = {
        "hyperparameters": {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "num_layers": num_layers,
            "fc_hidden_sizes": fc_hidden_sizes,
            "sequence_length": sequence_length
        },
        "gru": {"best_val_loss": best_gru_val_loss, "best_val_mae": best_gru_val_mae},
        "lstm": {
            "best_val_loss": best_lstm_val_loss,
            "best_val_mae": best_lstm_val_mae,
        },
    }

    with open(os.path.join(run_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Close the logger
    sys.stdout.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    batch_size=32,
    num_epochs=50,
    lr=0.005,
    num_layers=5,
    fc_hidden_sizes=[512, 256, 128, 64, 32],
    sequence_length=30
    
    main(
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        num_layers=num_layers,
        fc_hidden_sizes=fc_hidden_sizes,
        sequence_length=sequence_length
    )
