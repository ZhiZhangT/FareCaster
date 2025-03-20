# preprocessing.py
import pandas as pd
import numpy as np
import pickle
import os
import constants
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
    For each group, parse dates, create ordinal columns, and encode categorical features.

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
    # First, process each group individually (parse dates and create ordinal columns)
    processed_groups = {}
    for key, df in groups.items():
        df = df.copy()
        # Ensure that the date columns are datetime objects
        if not pd.api.types.is_datetime64_any_dtype(df["searchDate"]):
            df["searchDate"] = df["searchDate"].apply(parse_date)
        if not pd.api.types.is_datetime64_any_dtype(df["flightDate"]):
            df["flightDate"] = df["flightDate"].apply(parse_date)
        # Create ordinal columns
        df["searchDate_ordinal"] = df["searchDate"].apply(lambda x: x.toordinal())
        df["flightDate_ordinal"] = df["flightDate"].apply(lambda x: x.toordinal())
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


def preprocess_sample(raw_sample, le_start, le_dest, scaler):
    """
    Preprocess a raw sample for prediction.

    Parameters:
      raw_sample (array-like): A 2D array-like structure where each row contains:
          [searchDate, flightDate, startingAirport, destinationAirport, seatsRemaining]
      le_start (LabelEncoder): Pre-fitted LabelEncoder for 'startingAirport'.
      le_dest (LabelEncoder): Pre-fitted LabelEncoder for 'destinationAirport'.
      scaler (StandardScaler): Pre-fitted StandardScaler.

    Returns:
      processed_sample_scaled (np.ndarray): A scaled numpy array of shape (n_samples, 5).
    """
    processed_samples = []
    for row in raw_sample:
        search_date_ord = parse_date(row[0]).toordinal()
        flight_date_ord = parse_date(row[1]).toordinal()
        starting_airport_enc = le_start.transform([row[2]])[0]
        destination_airport_enc = le_dest.transform([row[3]])[0]
        seats_remaining = float(row[4])
        processed_samples.append(
            [
                search_date_ord,
                flight_date_ord,
                starting_airport_enc,
                destination_airport_enc,
                seats_remaining,
            ]
        )
    processed_samples = np.array(processed_samples, dtype=float)
    processed_sample_scaled = scaler.transform(processed_samples)
    return processed_sample_scaled


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


# Encapsulate all preprocessing steps into one function.
def preprocess_data(df):
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
    processed_groups = preprocess_grouped_dataframe(route_date_groups, fit=True)

    # Filter groups with at least 30 search dates
    filtered_groups = {
        key: group for key, group in processed_groups.items() if len(group) >= 30
    }

    # Define columns and parameters
    feature_cols = [
        "searchDate_ordinal",
        "flightDate_ordinal",
        "startingAirport_enc",
        "destinationAirport_enc",
        "seatsRemaining",
    ]
    target_col = "totalFare"
    sequence_length = 30

    sequences = []
    targets = []

    for key, group in filtered_groups.items():
        # sort the groups by searchDate
        group = group.sort_values("searchDate").reset_index(drop=True)
        # get the last 30 days of data
        window = group.iloc[-sequence_length:]

        # Compute ordinals if they don't exist
        if "searchDate_ordinal" not in window.columns:
            window["searchDate_ordinal"] = window["searchDate"].apply(
                lambda x: x.toordinal()
            )
        if "flightDate_ordinal" not in window.columns:
            window["flightDate_ordinal"] = window["flightDate"].apply(
                lambda x: x.toordinal()
            )

        sequences.append(window[feature_cols].values)
        targets.append(window[target_col].values[-1])

    X_all = np.array(sequences)
    y_all = np.array(targets)

    # Split into train, validation, and test sets.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.2, random_state=constants.RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=constants.RANDOM_STATE
    )

    # Scale the features.
    num_features = X_train.shape[2]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(
        X_train.shape
    )
    X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(
        X_val.shape
    )
    X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(
        X_test.shape
    )

    data = {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
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
    data = get_data()
    print("Train data shape:", data["X_train"].shape)
    print("Validation data shape:", data["X_val"].shape)
    print("Test data shape:", data["X_test"].shape)
