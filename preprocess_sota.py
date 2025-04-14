# preprocessing.py
import pandas as pd
import pickle
import os
import constants
import time
from sklearn.model_selection import train_test_split

# if saved directory does not exist, create it
if not os.path.exists(constants.SAVED_DIR):
    os.makedirs(constants.SAVED_DIR)

import pandas as pd


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
    group_by_cols = [
        "startingAirport",
        "destinationAirport",
        "flightDate",
        "searchDate",
    ]
    avg_cols = ["totalFare", "seatsRemaining"]

    # first_cols == cols that we simply take the first value
    # they refer to all columns that are not in avg_cols or the columns that we are grouping by
    first_cols = [
        col
        for col in dataframe.columns
        if not (col in avg_cols or col in group_by_cols)
    ]

    grouped_df = (
        dataframe.groupby(group_by_cols)
        .agg(
            {
                **{col: "mean" for col in avg_cols},
                **{col: "first" for col in first_cols},
            }
        )
        .reset_index()
    )

    # Group by route (startingAirport, destinationAirport, flightDate) regardless of searchDate
    route_date_groups = {
        key: group
        for key, group in grouped_df.groupby(
            ["startingAirport", "destinationAirport", "flightDate"]
        )
    }

    return route_date_groups


def format_columns(df):
    start_time = time.time()
    RENAME_MAPPING = {"searchDate": "ds", "totalFare": "y"}
    # Ensure that the date columns are datetime objects
    df["searchDate"] = df["searchDate"].apply(parse_date)
    df["flightDate"] = df["flightDate"].apply(parse_date)
    COLS_TO_KEEP = [
        "unique_id",
        "searchDate",
        "totalFare",
        "seatsRemaining",
        # NOTE: "flightDate" is dropped at the end of this function
        "flightDate",
        # TODO: uncomment this if we want to include totalTravelDistance
        # "totalTravelDistance",
        "segmentsDepartureTimeEpochSeconds",
    ]
    cols_to_keep_final = COLS_TO_KEEP.copy()
    for col in df.columns:
        if col.startswith("startingAirport_") or col.startswith("destinationAirport_"):
            cols_to_keep_final.append(col)

    df = df[cols_to_keep_final].copy()

    # day of week (0 = Monday, 6 = Sunday)
    df["searchDayOfWeek"] = df["searchDate"].dt.weekday
    df["flightDayOfWeek"] = df["flightDate"].dt.weekday

    # days between search and flight dates
    df["daysBetweenSearchAndFlight"] = (df["flightDate"] - df["searchDate"]).dt.days

    # month of search and flight dates
    df["searchMonth"] = df["searchDate"].dt.month
    df["flightMonth"] = df["flightDate"].dt.month

    # Convert the epoch seconds to a datetime column in UTC
    df["segmentsDepartureTime"] = pd.to_datetime(
        df["segmentsDepartureTimeEpochSeconds"], unit="s", utc=True
    )
    # hour that flight departed in UTC
    df["departureHourUTC"] = df["segmentsDepartureTime"].dt.hour

    # Rename columns
    df = df.rename(columns=RENAME_MAPPING)

    # drop columns that are not needed for the model
    df = df.drop(
        columns=[
            "flightDate",
            "segmentsDepartureTime",
            "segmentsDepartureTimeEpochSeconds",
        ]
    )

    print(f"Time taken to filter columns: {time.time() - start_time:.2f} seconds")
    return df


def one_hot_encode(df, col):
    start_time = time.time()
    # Factorize the column: returns a codes array (numerical representation)
    # and an Index of unique values (e.g., Index(['A', 'B', 'C']))
    codes, uniques = pd.factorize(df[col])

    # temporarily store the unique values in a new column
    df["airport_code"] = codes

    # One-hot encode the numeric column using get_dummies
    one_hot = pd.get_dummies(df["airport_code"], prefix=col)

    # convert from true/false to 1/0
    one_hot = one_hot.astype(int)

    # drop the temporary column
    df = df.drop(columns=["airport_code"])

    # Join the dummy columns back to the original dataframe
    df = df.join(one_hot)

    print(f"Time taken to one-hot encode {col}: {time.time() - start_time:.2f} seconds")
    return df


def split_dfs_into_halves(groups):
    """
    Splits each DataFrame in the provided list into two halves,
    and concatenates all first halves and second halves into two separate DataFrames.

    Parameters:
        groups (list of pd.DataFrame): List of grouped DataFrames to split.

    Returns:
        tuple: A tuple containing:
            - first_df (pd.DataFrame): Concatenation of the first halves of all dfs.
            - second_df (pd.DataFrame): Concatenation of the second halves of all dfs.
    """
    first_halves = []
    second_halves = []

    for group_df in groups:
        half_point = len(group_df) // 2  # Determine the split point
        first_halves.append(group_df.iloc[:half_point].copy())
        second_halves.append(group_df.iloc[half_point:].copy())

    first_df = (
        pd.concat(first_halves, ignore_index=True) if first_halves else pd.DataFrame()
    )
    second_df = (
        pd.concat(second_halves, ignore_index=True) if second_halves else pd.DataFrame()
    )

    return first_df, second_df


def preprocess_data(
    df,
    sequence_length,
):
    # TODO: uncomment this if we want to include totalTravelDistance
    # remove rows where totalTravelDistance is empty
    # df = df[df["totalTravelDistance"].notna()]

    df = one_hot_encode(df, "startingAirport")
    df = one_hot_encode(df, "destinationAirport")

    # Group routes by flight date
    route_date_groups = group_routes_by_flight_date(df)

    # Step 2: Filter out groups that don't meet the minimum sequence length
    filtered_groups = {
        k: v for k, v in route_date_groups.items() if len(v) >= sequence_length
    }

    # Step 3: For each group, sort by searchDate and create sliding window subgroups
    start_time = time.time()
    processed_groups = {}
    group_counter = 0
    for group_key, group_df in filtered_groups.items():
        group_key = "_".join(map(str, group_key))  # Create a unique key for the group
        group_df = group_df.copy()  # avoid SettingWithCopy warning
        group_df = group_df.sort_values("searchDate")  # Ensure rows are in order

        # If the group length equals the sequence_length, use the whole group
        if len(group_df) == sequence_length:
            unique_id = f"{group_key}_0"
            group_df["unique_id"] = unique_id
            processed_groups[unique_id] = group_df
            group_counter += 1
        # If the group has more rows than the sequence_length, split it using a sliding window
        elif len(group_df) > sequence_length:
            for i in range(len(group_df) - sequence_length + 1):
                unique_id = f"{group_key}_{i}"
                sub_df = group_df.iloc[i : i + sequence_length].copy()
                sub_df["unique_id"] = unique_id
                processed_groups[unique_id] = sub_df
                group_counter += 1
    print(f"Time taken to run sliding window: {time.time() - start_time:.2f} seconds")

    # Step 4: Shuffle groups and split them into train/test/validation splits (80/10/10)
    group_keys = list(processed_groups.keys())
    train_keys, temp_keys = train_test_split(
        group_keys, test_size=0.2, random_state=constants.RANDOM_STATE
    )
    # Then, split the temporary keys equally into validation and test (20% each of the total groups)
    val_keys, test_keys = train_test_split(
        temp_keys, test_size=0.5, random_state=constants.RANDOM_STATE
    )

    # Build lists of dataframes for each split based on the keys
    train_groups = [processed_groups[k] for k in train_keys]
    test_groups = [processed_groups[k] for k in test_keys]
    val_groups = [processed_groups[k] for k in val_keys]

    # Step 5: Flatten each split (concatenate all groups' dataframes)
    train_df = (
        pd.concat(train_groups, ignore_index=True) if train_groups else pd.DataFrame()
    )
    # Step 5b: For test and validation groups, split each processed group individually
    # so that "test_first" holds the first half and "test_second" holds the second half.
    test_first_df, test_second_df = split_dfs_into_halves(test_groups)
    val_first_df, val_second_df = split_dfs_into_halves(val_groups)

    all_dfs = [
        train_df,
        test_first_df,
        test_second_df,
        val_first_df,
        val_second_df,
    ]

    static_dfs = []
    for i, cur_df in enumerate(all_dfs):
        # Get the first occurrence for each unique value in "unique_id"
        unique_rows = cur_df.drop_duplicates(subset="unique_id", keep="first")

        # Determine the columns to keep
        cols_to_keep = []
        for col in unique_rows.columns:
            if col == "unique_id":
                cols_to_keep.append(col)
            elif col.startswith("startingAirport_") or col.startswith(
                "destinationAirport_"
            ):
                cols_to_keep.append(col)

        # Filter the DataFrame based on the selected columns
        filtered_df = unique_rows[cols_to_keep]
        static_dfs.append(filtered_df)

        # only keep columns relevant to model and not in static_df
        all_dfs[i] = format_columns(cur_df)

    # overwrite the original dfs with the formatted ones
    train_df, test_first_df, test_second_df, val_first_df, val_second_df = all_dfs

    # Concatenate all the filtered dfs
    # we drop duplicates again across all dfs just in case some unique_ids show up in more than one df
    static_df = pd.concat(static_dfs, ignore_index=True).drop_duplicates(
        subset="unique_id", keep="first"
    )

    # Step 7: Save the CSV files for each split
    train_df.to_csv(f"{constants.SAVED_DIR}/train.csv", index=False)
    test_first_df.to_csv(f"{constants.SAVED_DIR}/test_first.csv", index=False)
    test_second_df.to_csv(f"{constants.SAVED_DIR}/test_second.csv", index=False)
    val_first_df.to_csv(f"{constants.SAVED_DIR}/validation_first.csv", index=False)
    val_second_df.to_csv(f"{constants.SAVED_DIR}/validation_second.csv", index=False)
    static_df.to_csv(f"{constants.SAVED_DIR}/static.csv", index=False)

    return {
        "X_train": train_df,
        "X_test_first": test_first_df,
        "X_test_second": test_second_df,
        "X_val_first": val_first_df,
        "X_val_second": val_second_df,
        "static": static_df,
    }


# Save processed data and fitted scaler to disk.
def save_processed_data(data, data_filepath):
    with open(data_filepath, "wb") as f:
        pickle.dump(data, f)


# Load processed data and scaler from disk.
def load_processed_data(data_filepath):
    with open(data_filepath, "rb") as f:
        data = pickle.load(f)

    return data


def get_last_time_steps(data, time_steps):
    """
    Slices the second dimension of each data array to keep only the last {time_steps} time steps.
    """
    data_sliced = {}
    for key, array in data.items():
        # Ensure the array has at least two dimensions and enough time steps
        if array.ndim >= 2 and array.shape[1] >= time_steps:
            # used for features (e.g. X_train)
            if array.ndim == 3:
                data_sliced[key] = array[:, -time_steps:, :]
            # used for target (e.g. y_train)
            elif array.ndim == 2:
                data_sliced[key] = array[:, -time_steps:]
            else:
                raise ValueError(f"Unexpected array shape for key {key}: {array.shape}")
        else:
            data_sliced[key] = array
    return data_sliced


def load_raw_data(filepath=constants.RAW_DATA_PATH):
    return pd.read_csv(filepath)


# Main function to get data. It will load from cache if available unless forced to reprocess.
def get_data(data_filepath, force_reprocess=False, sequence_length=30, time_steps=None):
    if not force_reprocess and os.path.exists(data_filepath):
        print("Loading preprocessed data from disk...")
        data = load_processed_data(data_filepath=data_filepath)
    else:
        print("Processing raw data...")
        df = load_raw_data()
        start_time = time.time()
        data = preprocess_data(df, sequence_length=sequence_length)
        print(f"Time taken to preprocess data: {time.time() - start_time:.2f} seconds")
        save_processed_data(data, data_filepath)

    if time_steps is not None:
        half_sequence_length = sequence_length // 2
        if time_steps > half_sequence_length:
            raise ValueError(
                f"time_steps should be less than or equal to half of sequence_length ({half_sequence_length})."
            )
        data_sliced = get_last_time_steps(data, time_steps)
        return data_sliced
    else:
        return data


def save_to_csv(arr, filename):
    """
    Reshape an array of shape (n_samples, sequence_length, 1) to (n_samples, sequence_length)
    and save it as a CSV file.
    """
    # Check if the array has a singleton third dimension and remove it
    if arr.ndim == 3 and arr.shape[2] == 1:
        reshaped = arr.reshape(arr.shape[0], arr.shape[1])
    else:
        reshaped = arr
    # Save the reshaped array to CSV without row indices
    pd.DataFrame(reshaped).to_csv(filename, index=False)


if __name__ == "__main__":
    # This will run the full pipeline only if cached files are missing.
    data = get_data(
        data_filepath=constants.PROCESSED_DATA_FILE_SOTA, force_reprocess=True
    )

    train_df = data["X_train"]
    test_first_df = data["X_test_first"]
    test_second_df = data["X_test_second"]
    val_first_df = data["X_val_first"]
    val_second_df = data["X_val_second"]
    static_df = data["static"]

    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_first_df.shape}")
    print(f"Test data second shape: {test_second_df.shape}")
    print(f"Validation data shape: {val_first_df.shape}")
    print(f"Validation data second shape: {val_second_df.shape}")
    # EXPECTED: 4898100 / 30 + (306135 * 2 / 15) == 204088
    print(f"Static data shape: {static_df.shape}")

# WITH TOTAL TRAVEL DISTANCE
# Train data shape: (4552080, 14); 151736
# Test data shape: (284520, 14)
# Validation data shape: (284505, 14)
# Test data second shape: (284520, 14)
# Validation data second shape: (284505, 14)

# WITHOUT TOTAL TRAVEL DISTANCE
# Train data shape: (4898100, 13); 163270
# Test data shape: (306135, 13)
# Validation data shape: (306135, 13)
# Test data second shape: (306135, 13)
# Validation data second shape: (306135, 13)
