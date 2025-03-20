import torch
import torch.nn as nn
import time
from models import GRUModel, LSTMModel
from torch.utils.data import TensorDataset, DataLoader
from preprocess import get_data


# -------------------------------
# Load and preprocess the dataset
# -------------------------------

# data = {
#     "X_train": X_train_scaled,
#     "y_train": y_train,
#     "X_val": X_val_scaled,
#     "y_val": y_val,
#     "X_test": X_test_scaled,
#     "y_test": y_test,
# }


data = get_data()

X_train_scaled = data["X_train"]
y_train = data["y_train"]
X_val_scaled = data["X_val"]
y_val = data["y_val"]
X_test_scaled = data["X_test"]
y_test = data["y_test"]


# ----- Convert to PyTorch Tensors -----
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create DataLoaders with shuffling disabled (to preserve sorted order)
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# -------------------------------
# Training and Evaluation Functions
# -------------------------------


def train_model(
    model, train_loader, val_loader, num_epochs=10, lr=0.001, model_save_path=None
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

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_epoch = -1

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

    return model


def evaluate_model(model, test_loader):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Evaluating on device: {device}")
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()
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
# Train and Evaluate the Models
# -------------------------------

input_size = X_train_scaled.shape[2]  # number of features
print(f"input_size (number of features): {input_size}")

# Train GRU Model
print("Training GRU model...")
gru_model = GRUModel(
    input_size=input_size,
)
gru_save_path = "best_gru_model.pt"
gru_model = train_model(
    gru_model,
    train_loader,
    val_loader,
    num_epochs=50,
    lr=0.005,
    model_save_path=gru_save_path,
)
gru_test_loss, gru_test_mae = evaluate_model(gru_model, test_loader)
print(f"GRU Model Test Loss: {gru_test_loss:.4f}, Test MAE: {gru_test_mae:.4f}")

# Train LSTM Model
print("\nTraining LSTM model...")
lstm_model = LSTMModel(
    input_size=input_size,
)
lstm_save_path = "best_lstm_model.pt"
lstm_model = train_model(
    lstm_model,
    train_loader,
    val_loader,
    num_epochs=50,
    lr=0.005,
    model_save_path=lstm_save_path,
)
lstm_test_loss, lstm_test_mae = evaluate_model(lstm_model, test_loader)
print(f"LSTM Model Test Loss: {lstm_test_loss:.4f}, Test MAE: {lstm_test_mae:.4f}")
