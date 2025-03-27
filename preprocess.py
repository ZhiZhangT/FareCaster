# preprocessing.py
import pandas as pd
import numpy as np
import pickle
import os
import constants
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# if saved directory does not exist, create it
if not os.path.exists(constants.SAVED_DIR):
    os.makedirs(constants.SAVED_DIR)


def parse_date(date_str):
    """
    Parse a date string using several common formats.
    """
    for fmt in ("%d/%m/%y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"No valid date format found for: {date_str}")


def preprocess_grouped_dataframe(groups, fit=True, le_start=None, le_dest=None):
    """
    Preprocess a dictionary of DataFrame groups (output of group_routes_by_flight_date).
    For each group, parse dates, and encode categorical features.

    Parameters:
      groups: dict
          Dictionary where keys are group identifiers and values are DataFrames.
      fit: bool
          Whether to fit new LabelEncoders or use the provided ones.
      le_start: LabelEncoder or None
          Pre-fitted label encoder for 'startingAirport' (if fit is False).
      le_dest: LabelEncoder or None
          Pre-fitted label encoder for 'destinationAirport' (if fit is False).

    Returns:
      processed_groups: dict
          Dictionary of processed DataFrames with additional columns.
      (le_start, le_dest): tuple
          The fitted label encoders.
    """
    processed_groups = {}
    for key, df in groups.items():
        df = df.copy()
        # Ensure that the date columns are datetime objects
        if not pd.api.types.is_datetime64_any_dtype(df["searchDate"]):
            df["searchDate"] = df["searchDate"].apply(parse_date)
        if not pd.api.types.is_datetime64_any_dtype(df["flightDate"]):
            df["flightDate"] = df["flightDate"].apply(parse_date)
        # check if the column "segmentsDepartureTime" exists
        if "segmentsDepartureTime" not in df.columns:
            df["segmentsDepartureTime"] = pd.to_datetime(
                df["segmentsDepartureTimeEpochSeconds"], unit="s"
            )

        # day of week (0 = Monday, 6 = Sunday)
        df["searchDayOfWeek"] = df["searchDate"].dt.weekday
        df["flightDayOfWeek"] = df["flightDate"].dt.weekday

        # days between search and flight dates
        df["daysBetweenSearchAndFlight"] = (df["flightDate"] - df["searchDate"]).dt.days

        # month of search and flight dates
        df["searchMonth"] = df["searchDate"].dt.month
        df["flightMonth"] = df["flightDate"].dt.month

        # hour that flight departed in UTC
        df["departureHourUTC"] = df["segmentsDepartureTime"].dt.hour
        processed_groups[key] = df

    # Combine all groups to fit the LabelEncoders on the complete data
    combined_df = pd.concat(processed_groups.values(), ignore_index=True)

    # Encode categorical features: startingAirport and destinationAirport
    if fit:
        le_start = LabelEncoder()
        le_dest = LabelEncoder()
        le_start.fit(combined_df["startingAirport"])
        le_dest.fit(combined_df["destinationAirport"])
        with open(f"{constants.SAVED_DIR}le_start.pkl", "wb") as f:
            pickle.dump(le_start, f)

        with open(f"{constants.SAVED_DIR}le_dest.pkl", "wb") as f:
            pickle.dump(le_dest, f)
    else:
        if le_start is None or le_dest is None:
            raise ValueError(
                "Pre-fitted label encoders must be provided when fit=False."
            )

    # Apply encoding to each group using the fitted encoders
    for key, df in processed_groups.items():
        df["startingAirport_enc"] = le_start.transform(df["startingAirport"])
        df["destinationAirport_enc"] = le_dest.transform(df["destinationAirport"])
        processed_groups[key] = df

    return processed_groups


def group_routes_by_flight_date(dataframe):
    """
    Groups the dataframe by startingAirport, destinationAirport, and flightDate.
    Returns a dictionary where keys are (startingAirport, destinationAirport, flightDate) tuples
    and values are the corresponding grouped dataframes.

    Parameters:
    -----------
    dataframe : pandas.DataFrame
        The input dataframe containing flight data

    Returns:
    --------
    dict
        Dictionary of grouped dataframes
    """
    # If there are multiple rows with the same searchDate in a group, pool them together by taking the mean
    # For the rest of the columns, we can take the first value since they should be the same
    # fareBasisCode                                 object
    # elapsedDays                                    int64
    # isBasicEconomy                                  bool
    # isRefundable                                    bool
    # isNonStop                                       bool
    # baseFare                                     float64
    # totalFare                                    float64
    # seatsRemaining                                 int64
    # totalTravelDistance                          float64
    # segmentsDepartureTimeEpochSeconds              int64
    # segmentsArrivalTimeEpochSeconds                int64
    # segmentsAirlineCode                           object
    # segmentsEquipmentDescription                  object
    # segmentsDurationInSeconds                      int64
    # segmentsCabinCode                             object
    # segmentsDepartureTime                 datetime64[ns]
    # segmentsArrivalTime                   datetime64[ns]
    # segmentsDuration                     timedelta64[ns]

    # Get columns that need to be averaged
    avg_cols = ["baseFare", "totalFare", "seatsRemaining"]

    # Get columns that need to be taken from the first row
    first_cols = [
        "fareBasisCode",
        "elapsedDays",
        "isBasicEconomy",
        "isRefundable",
        "isNonStop",
        "totalTravelDistance",
        "segmentsAirlineCode",
        "segmentsEquipmentDescription",
        "segmentsCabinCode",
        "segmentsDepartureTime",
        "segmentsArrivalTime",
        "segmentsDuration",
    ]

    grouped_df = (
        dataframe.groupby(
            ["startingAirport", "destinationAirport", "flightDate", "searchDate"]
        )
        .agg(
            {
                **{col: "mean" for col in avg_cols},
                **{col: "first" for col in first_cols},
            }
        )
        .reset_index()
    )

    # Convert to dictionary structure
    route_date_groups = {
        key: group
        for key, group in grouped_df.groupby(
            ["startingAirport", "destinationAirport", "flightDate"]
        )
    }

    return route_date_groups


def parse_date(date_str):
    for fmt in ("%d/%m/%y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"No valid date format found for: {date_str}")


# Wrap your raw data loading in a function.
def load_raw_data(filepath=constants.RAW_DATA_PATH):
    return pd.read_csv(filepath)


# Function to perform cyclical encoding on the cyclical features.
def cyclical_encode(X, feature_cols, cyclical_features):
    """
    For each cyclical feature, compute sine and cosine values and append them.
    Then remove the original cyclical columns from X.
    """
    indices_to_remove = []
    new_features = []
    for feature, period in cyclical_features.items():
        idx = feature_cols.index(feature)
        indices_to_remove.append(idx)
        # Extract the feature values (shape: [samples, sequence_length])
        values = X[..., idx]
        # Compute sine and cosine transformations.
        sin_feat = np.sin(2 * np.pi * values / period)
        cos_feat = np.cos(2 * np.pi * values / period)
        # Expand dims so they can be concatenated along the last axis.
        sin_feat = np.expand_dims(sin_feat, axis=-1)
        cos_feat = np.expand_dims(cos_feat, axis=-1)
        new_features.append(sin_feat)
        new_features.append(cos_feat)

    # Remove the original cyclical columns.
    X_updated = np.delete(X, indices_to_remove, axis=-1)
    # Append the new sine and cosine features.
    X_updated = np.concatenate([X_updated] + new_features, axis=-1)
    return X_updated


def get_last_window(group, seq_len, feature_cols, target_col):
    group = group.sort_values("searchDate").reset_index(drop=True)
    if len(group) < seq_len:
        return None, None
    window = group.iloc[-seq_len:]
    return window[feature_cols].values, window[target_col].values[-1]


def get_sliding_windows(group, seq_len, feature_cols, target_col):
    group = group.sort_values("searchDate").reset_index(drop=True)
    windows, targets = [], []
    if len(group) <= seq_len:
        return windows, targets
    # Create a sliding window for each possible position.
    for i in range(seq_len, len(group)):
        # first group starts from index 0 to seq_len
        window = group.iloc[i - seq_len : i]
        windows.append(window[feature_cols].values)
        targets.append(group.iloc[i][target_col])
    return windows, targets


def create_datasets(keys, filtered_groups, sequence_length, feature_cols, target_col):
    X_last, y_last = [], []
    X_slide, y_slide = [], []
    for key in keys:
        group = filtered_groups[key]
        last_seq, last_target = get_last_window(
            group, sequence_length, feature_cols, target_col
        )
        if last_seq is not None:
            X_last.append(last_seq)
            y_last.append(last_target)
        windows, targets = get_sliding_windows(
            group, sequence_length, feature_cols, target_col
        )
        X_slide.extend(windows)
        y_slide.extend(targets)
    return np.array(X_last), np.array(y_last), np.array(X_slide), np.array(y_slide)


# Encapsulate all preprocessing steps into one function.
def preprocess_data(
    df,
    feature_cols=constants.FEATURE_COLS,
    categorical_cols=constants.CATEGORICAL_COLS,
    cyclical_features=constants.CYCLICAL_FEATURES,
):
    # NOTE: these fields have to be added before group_routes_by_flight_date as they are used in the function
    # Convert epoch seconds to datetime
    df["segmentsDepartureTime"] = pd.to_datetime(
        df["segmentsDepartureTimeEpochSeconds"], unit="s"
    )
    df["segmentsArrivalTime"] = pd.to_datetime(
        df["segmentsArrivalTimeEpochSeconds"], unit="s"
    )

    # Convert duration to timedelta
    df["segmentsDuration"] = pd.to_timedelta(df["segmentsDurationInSeconds"], unit="s")

    # Group routes by flight date
    route_date_groups = group_routes_by_flight_date(df)
    start_time = time.time()
    processed_groups = preprocess_grouped_dataframe(route_date_groups, fit=True)
    print(
        f"Time taken to run preprocess_grouped_dataframe(): {time.time() - start_time:.2f} seconds"
    )

    # Filter groups with at least 30 search dates
    filtered_groups = {
        key: group for key, group in processed_groups.items() if len(group) >= 30
    }

    target_col = "totalFare"
    sequence_length = 30

    # --- Split into train/test/validation first ---

    group_keys = list(filtered_groups.keys())
    train_keys, temp_keys = train_test_split(
        group_keys, test_size=0.2, random_state=constants.RANDOM_STATE
    )
    val_keys, test_keys = train_test_split(
        temp_keys, test_size=0.5, random_state=constants.RANDOM_STATE
    )

    # Create datasets for each split.
    X_train, y_train, X_train_sliding, y_train_sliding = create_datasets(
        train_keys, filtered_groups, sequence_length, feature_cols, target_col
    )
    X_val, y_val, X_val_sliding, y_val_sliding = create_datasets(
        val_keys, filtered_groups, sequence_length, feature_cols, target_col
    )
    X_test, y_test, X_test_sliding, y_test_sliding = create_datasets(
        test_keys, filtered_groups, sequence_length, feature_cols, target_col
    )

    # --- Helper Function for Scaling ---

    # Identify numeric columns (i.e. features not in categorical_cols).
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    numeric_indices = [feature_cols.index(col) for col in numeric_cols]

    scaler = StandardScaler()

    def scale_data(X, scaler, numeric_indices, fit=False):
        # Flatten the numeric features
        X_numeric = X[..., numeric_indices].reshape(-1, len(numeric_indices))
        if fit:
            X_numeric_scaled = scaler.fit_transform(X_numeric)
        else:
            X_numeric_scaled = scaler.transform(X_numeric)
        # Reshape back to the original 3D shape and replace numeric features.
        X_numeric_scaled = X_numeric_scaled.reshape(
            X.shape[0], X.shape[1], len(numeric_indices)
        )
        X_scaled = X.copy()
        X_scaled[..., numeric_indices] = X_numeric_scaled
        return X_scaled

    # --- Scaling and Cyclical Encoding ---

    X_train_scaled = scale_data(X_train, scaler, numeric_indices, fit=True)
    X_val_scaled = scale_data(X_val, scaler, numeric_indices)
    X_test_scaled = scale_data(X_test, scaler, numeric_indices)

    X_train_final = cyclical_encode(X_train_scaled, feature_cols, cyclical_features)
    X_val_final = cyclical_encode(X_val_scaled, feature_cols, cyclical_features)
    X_test_final = cyclical_encode(X_test_scaled, feature_cols, cyclical_features)

    X_train_sliding_scaled = scale_data(X_train_sliding, scaler, numeric_indices)
    X_val_sliding_scaled = scale_data(X_val_sliding, scaler, numeric_indices)
    X_test_sliding_scaled = scale_data(X_test_sliding, scaler, numeric_indices)

    X_train_sliding_final = cyclical_encode(
        X_train_sliding_scaled, feature_cols, cyclical_features
    )
    X_val_sliding_final = cyclical_encode(
        X_val_sliding_scaled, feature_cols, cyclical_features
    )
    X_test_sliding_final = cyclical_encode(
        X_test_sliding_scaled, feature_cols, cyclical_features
    )

    # --- Package the Data ---

    data = {
        "X_train_scaled": X_train_final,
        "y_train": y_train,
        "X_val_scaled": X_val_final,
        "y_val": y_val,
        "X_test_scaled": X_test_final,
        "y_test": y_test,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "X_train_sliding_window_scaled": X_train_sliding_final,
        "y_train_sliding_window": y_train_sliding,
        "X_val_sliding_window_scaled": X_val_sliding_final,
        "y_val_sliding_window": y_val_sliding,
        "X_test_sliding_window_scaled": X_test_sliding_final,
        "y_test_sliding_window": y_test_sliding,
    }

    with open(constants.SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    return data


# Save processed data and fitted scaler to disk.
def save_processed_data(data, data_filepath=constants.PROCESSED_DATA_FILE):
    with open(data_filepath, "wb") as f:
        pickle.dump(data, f)


# Load processed data and scaler from disk.
def load_processed_data(data_filepath=constants.PROCESSED_DATA_FILE):
    with open(data_filepath, "rb") as f:
        data = pickle.load(f)

    return data


# Main function to get data. It will load from cache if available unless forced to reprocess.
def get_data(force_reprocess=False):
    if (
        not force_reprocess
        and os.path.exists(constants.PROCESSED_DATA_FILE)
        and os.path.exists(constants.SCALER_FILE)
    ):
        print("Loading preprocessed data and scaler from disk...")
        return load_processed_data()
    else:
        print("Processing raw data...")
        df = load_raw_data()
        data = preprocess_data(df)
        save_processed_data(data)
        return data


if __name__ == "__main__":
    # This will run the full pipeline only if cached files are missing.
    data = get_data(force_reprocess=False)
    print("Train data shape:", data["X_train_scaled"].shape)
    print("Validation data shape:", data["X_val_scaled"].shape)
    print("Test data shape:", data["X_test_scaled"].shape)
    print(
        f"Train data for sliding window: {data['X_train_sliding_window_scaled'].shape}"
    )
    print(
        f"Validation data for sliding window: {data['X_val_sliding_window_scaled'].shape}"
    )
    print(f"Test data for sliding window: {data['X_test_sliding_window_scaled'].shape}")
