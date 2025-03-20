import pandas as pd
import torch
import torch.nn as nn
import constants
import time
from models import GRUModel, LSTMModel
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from preprocess import preprocess_dataframe
from sklearn.preprocessing import StandardScaler


# -------------------------------
# Load and preprocess the dataset
# -------------------------------

# Load CSV file (adjust the path if necessary)
df = pd.read_csv("data/itineraries_filtered.csv")

# -------------------------------
# Split the original dataset (raw, unprocessed) into train (80%), validation (10%), and test (10%)
# -------------------------------
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=constants.RANDOM_STATE
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=constants.RANDOM_STATE
)


# Save raw validation and test datasets to CSV (all columns preserved)
val_df.to_csv(constants.VALIDATION_FILEPATH, index=False)
test_df.to_csv(constants.TEST_FILEPATH, index=False)


def parse_date(date_str):
    # List of possible date formats
    for fmt in ("%d/%m/%y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    # If none of the formats match, raise an error
    raise ValueError(f"No valid date format found for: {date_str}")


# # Save the fitted scaler and label encoders
# with open("scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# with open("le_start.pkl", "wb") as f:
#     pickle.dump(le_start, f)

# with open("le_dest.pkl", "wb") as f:
#     pickle.dump(le_dest, f)


# Preprocess the full DataFrame
processed_df, (le_start, le_dest) = preprocess_dataframe(df, fit=True)

# Specify your target column
target_col = "totalFare"

# First split: 80% train and 20% temporary
df_train, df_temp = train_test_split(
    processed_df, test_size=0.2, random_state=constants.RANDOM_STATE
)
# Then split the temporary set equally into validation and test (10% each)
df_val, df_test = train_test_split(
    df_temp, test_size=0.5, random_state=constants.RANDOM_STATE
)

# Sorting on the original date columns (which are still in datetime format)
df_train = df_train.sort_values(by=["searchDate", "flightDate"])
df_val = df_val.sort_values(by=["searchDate", "flightDate"])
df_test = df_test.sort_values(by=["searchDate", "flightDate"])

# Define the feature columns to use (use the processed/ordinal/encoded columns)
feature_cols = [
    "searchDate_ordinal",
    "flightDate_ordinal",
    "startingAirport_enc",
    "destinationAirport_enc",
    "seatsRemaining",
]

# Extract features and target values from each split
X_train_df = df_train[feature_cols]
y_train = df_train[target_col].values

X_val_df = df_val[feature_cols]
y_val = df_val[target_col].values

X_test_df = df_test[feature_cols]
y_test = df_test[target_col].values

# Fit the scaler on training features and transform validation and test features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_val_scaled = scaler.transform(X_val_df)
X_test_scaled = scaler.transform(X_test_df)

# Reshape to (samples, timesteps, features)
X_train_scaled = X_train_scaled.reshape(
    (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
)
X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape(
    (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
)

# Convert arrays to PyTorch tensors
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# Training and Evaluation Functions
# -------------------------------


def train_model(
    model, train_loader, val_loader, num_epochs=10, lr=0.001, model_save_path=None
):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
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

    # Optionally, load the best model state at the end of training
    if best_epoch != -1 and model_save_path is not None:
        model.load_state_dict(torch.load(model_save_path))
        print(
            f"Loaded best model from epoch {best_epoch+1} with val loss {best_val_loss:.4f}"
        )

    print(f"Time taken for epoch {num_epochs}: {time.time() - start_time:.2f}s")

    return model


def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
print(f"input_size: {input_size}")

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
    lr=0.001,
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
    lr=0.001,
    model_save_path=lstm_save_path,
)
lstm_test_loss, lstm_test_mae = evaluate_model(lstm_model, test_loader)
print(f"LSTM Model Test Loss: {lstm_test_loss:.4f}, Test MAE: {lstm_test_mae:.4f}")
