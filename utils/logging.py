import math
import torch
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import json


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
    # ensure that the directory exists
    os.makedirs(save_dir, exist_ok=True)
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
