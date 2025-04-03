import math
import torch
import torch.nn as nn
import time
import os
import sys
from datetime import datetime
from models import GRUModel, LSTMModel, GRUModelDeep, LSTMModelDeep, TransformerModel
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
        return x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)

    return torch.mean(_log_cosh(y_pred - y_true))


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
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
    elif torch.backends.mps.is_available():
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
    elif torch.backends.mps.is_available():
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


# -------------------------------
# Main execution flow with run directory
# -------------------------------


def main(
    models,
    summary,
    use_sliding_window,
    sequence_length,
    batch_size,
    num_epochs,
    lr,
    train_loss_criteria,
    val_loss_criteria,
):
    if "hyperparameters" not in summary:
        raise ValueError("Summary must contain 'hyperparameters' key.")
    if "common" not in summary["hyperparameters"]:
        raise ValueError("Summary['hyperparameters'] must contain 'common' key.")

    # Load and preprocess data
    data = get_data(sequence_length=sequence_length)

    if use_sliding_window:
        X_train_scaled = data["X_train_sliding_window_scaled"]
        y_train = data["y_train_sliding_window"]
        X_val_scaled = data["X_val_sliding_window_scaled"]
        y_val = data["y_val_sliding_window"]
    else:
        X_train_scaled = data["X_train_scaled"]
        y_train = data["y_train"]
        X_val_scaled = data["X_val_scaled"]
        y_val = data["y_val"]

    print(f"Train data shape: {X_train_scaled.shape}, {y_train.shape}")
    print(f"Validation data shape: {X_val_scaled.shape}, {y_val.shape}")

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_size = X_train_scaled.shape[2]  # number of features
    print(f"input_size (number of features): {input_size}")

    results = {}
    for model_name, model_info in models.items():
        print(f"\nTraining {model_name} model...")
        # initialise the model with custom parameters
        # NOTE: this code is equivalent to gru_model = GRUModelDeep(input_size=input_size, ...); but it can work with any model in the models dict
        model = model_info["class"](input_size=input_size, **model_info["params"])
        model, losses = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            lr=lr,
            save_dir=run_dir,
            model_type=model_name.lower(),
            loss_criteria=train_loss_criteria,
        )
        plot_losses(losses, run_dir, model_name.lower())
        best_val_loss, best_val_mae = evaluate_model(
            model, val_loader, val_loss_criteria
        )
        print(
            f"BEST {model_name} Model Val Loss: {best_val_loss:.4f}, Val MAE: {best_val_mae:.4f}"
        )
        results[model_name] = {
            "best_val_loss": best_val_loss,
            "best_val_mae": best_val_mae,
        }

    # save experiment results on validation set
    summary["results"] = results

    # save model-specific hyperparameters
    for model_name, model_info in models.items():
        summary["hyperparameters"][model_name] = model_info["params"]

    with open(os.path.join(run_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Plot comparative bar chart for val MAE
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    mae_values = [results[m]["best_val_mae"] for m in model_names]
    # Use a subset of colors based on the number of models
    colors = ["blue", "green", "purple"][: len(model_names)]
    plt.bar(model_names, mae_values, color=colors)
    plt.title("Model Comparison: Val MAE")
    plt.ylabel("Mean Absolute Error")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(os.path.join(run_dir, "model_comparison.png"))
    plt.close()


if __name__ == "__main__":
    batch_size = 32
    num_epochs = 50
    lr = 0.005
    num_layers = 5
    fc_hidden_sizes = [512, 256, 128, 64, 32]
    sequence_length = 30
    use_sliding_window = True
    train_loss_criteria = nn.MSELoss()
    val_loss_criteria = nn.MSELoss()
    run_name = "DeepModels_SlidingWindow_LogCoshLossTrain_MSELossVal"
    # Transformer specific parameters
    transformer_hidden_size = 64
    transformer_nhead = 4
    transformer_dropout = 0.1

    # Create run directory
    run_dir = create_run_directory(run_name)

    # Set up logging to capture all print statements
    sys.stdout = Logger(os.path.join(run_dir, "training_log.txt"))

    print(f"Starting training run in directory: {run_dir}")

    # Log hyperparameters
    print("Hyperparameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_epochs: {num_epochs}")
    print(f"  learning rate: {lr}")
    print(f"  num_layers: {num_layers}")
    print(f"  fc_hidden_sizes: {fc_hidden_sizes}")
    print(f"  sequence_length: {sequence_length}")
    print(f"  use_sliding_window: {use_sliding_window}")
    print(f"  transformer_hidden_size: {transformer_hidden_size}")
    print(f"  transformer_nhead: {transformer_nhead}")
    print(f"  transformer_dropout: {transformer_dropout}")

    # save common hyperparameters
    summary = {
        "hyperparameters": {
            "common": {
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": lr,
                "sequence_length": sequence_length,
                "use_sliding_window": use_sliding_window,
            },
        }
    }

    # Dictionary for models and model-specific parameters
    models = {
        "GRU": {
            "class": GRUModelDeep,
            "params": {
                "num_layers": num_layers,
                "fc_hidden_sizes": fc_hidden_sizes,
            },
        },
        "LSTM": {
            "class": LSTMModelDeep,
            "params": {
                "num_layers": num_layers,
                "fc_hidden_sizes": fc_hidden_sizes,
            },
        },
        "Transformer": {
            "class": TransformerModel,
            "params": {
                "hidden_size": transformer_hidden_size,
                "num_layers": min(
                    3, num_layers
                ),  # Transformers often need fewer layers
                "nhead": transformer_nhead,
                "dropout": transformer_dropout,
            },
        },
    }

    main(
        models,
        summary=summary,
        use_sliding_window=use_sliding_window,
        sequence_length=sequence_length,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        train_loss_criteria=train_loss_criteria,
        val_loss_criteria=val_loss_criteria,
    )

    # Close the logger
    sys.stdout.close()
    sys.stdout = sys.__stdout__
