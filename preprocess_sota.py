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
    avg_cols = ["baseFare", "totalFare", "seatsRemaining"]

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

    # Group by route (startingAirport, destinationAirport, flightDate) regardless of searchDate
    route_date_groups = {
        key: group
        for key, group in grouped_df.groupby(
            ["startingAirport", "destinationAirport", "flightDate"]
        )
    }

    return route_date_groups


def preprocess_data(
    df,
    sequence_length,
):

    # Ensure that the date columns are datetime objects
    if not pd.api.types.is_datetime64_any_dtype(df["searchDate"]):
        df["searchDate"] = df["searchDate"].apply(parse_date)

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
    test_first_halves = []
    test_second_halves = []
    for group_df in test_groups:
        half_point = len(group_df) // 2  # Determine the split point
        first_half = group_df.iloc[:half_point].copy()
        second_half = group_df.iloc[half_point:].copy()
        # Append a suffix to differentiate the halves
        first_half["unique_id"] = first_half["unique_id"] + "_first"
        second_half["unique_id"] = second_half["unique_id"] + "_second"
        test_first_halves.append(first_half)
        test_second_halves.append(second_half)
    test_first_df = (
        pd.concat(test_first_halves, ignore_index=True)
        if test_first_halves
        else pd.DataFrame()
    )
    test_second_df = (
        pd.concat(test_second_halves, ignore_index=True)
        if test_second_halves
        else pd.DataFrame()
    )

    val_first_halves = []
    val_second_halves = []
    for group_df in val_groups:
        half_point = len(group_df) // 2  # Determine the split point
        first_half = group_df.iloc[:half_point].copy()
        second_half = group_df.iloc[half_point:].copy()
        first_half["unique_id"] = first_half["unique_id"] + "_first"
        second_half["unique_id"] = second_half["unique_id"] + "_second"
        val_first_halves.append(first_half)
        val_second_halves.append(second_half)
    val_first_df = (
        pd.concat(val_first_halves, ignore_index=True)
        if val_first_halves
        else pd.DataFrame()
    )
    val_second_df = (
        pd.concat(val_second_halves, ignore_index=True)
        if val_second_halves
        else pd.DataFrame()
    )

    # Step 6: Rename columns and drop unnecessary columns
    rename_mapping = {"searchDate": "ds", "totalFare": "y"}
    drop_cols = [
        "startingAirport",
        "destinationAirport",
        "flightDate",
        "baseFare",
        "seatsRemaining",
        "fareBasisCode",
        "elapsedDays",
        "isBasicEconomy",
        "isRefundable",
        "isNonStop",
        "totalTravelDistance",
        "segmentsAirlineCode",
        "segmentsEquipmentDescription",
        "segmentsCabinCode",
    ]

    train_df_condensed = train_df.rename(columns=rename_mapping).drop(columns=drop_cols)
    test_first_df_condensed = test_first_df.rename(columns=rename_mapping).drop(columns=drop_cols)
    test_second_df_condensed = test_second_df.rename(columns=rename_mapping).drop(columns=drop_cols)
    val_first_df_condensed = val_first_df.rename(columns=rename_mapping).drop(columns=drop_cols)
    val_second_df_condensed = val_second_df.rename(columns=rename_mapping).drop(columns=drop_cols)

    # Step 7: Save the CSV files for each split
    train_df_condensed.to_csv(f"{constants.SAVED_DIR}/train.csv", index=False)
    test_first_df_condensed.to_csv(f"{constants.SAVED_DIR}/test_first.csv", index=False)
    test_second_df_condensed.to_csv(f"{constants.SAVED_DIR}/test_second.csv", index=False)
    val_first_df_condensed.to_csv(f"{constants.SAVED_DIR}/validation_first.csv", index=False)
    val_second_df_condensed.to_csv(f"{constants.SAVED_DIR}/validation_second.csv", index=False)

    return {
        "X_train": train_df,
        "X_test_first": test_first_df,
        "X_test_second": test_second_df,
        "X_val_first": val_first_df,
        "X_val_second": val_second_df,
    }


# Save processed data and fitted scaler to disk.
def save_processed_data(data, data_filepath=constants.PROCESSED_DATA_FILE):
    with open(data_filepath, "wb") as f:
        pickle.dump(data, f)


# Load processed data and scaler from disk.
def load_processed_data(data_filepath=constants.PROCESSED_DATA_FILE):
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
def get_data(force_reprocess=False, sequence_length=30, time_steps=None):
    if not force_reprocess and os.path.exists(constants.PROCESSED_DATA_FILE):
        print("Loading preprocessed data from disk...")
        data = load_processed_data()
    else:
        print("Processing raw data...")
        df = load_raw_data()
        data = preprocess_data(df, sequence_length=sequence_length)
        save_processed_data(data)

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
    data = get_data(force_reprocess=False)
    print(f"Train data shape: {data['X_train'].shape}")
    print(f"Test data shape: {data['X_test_first'].shape}")
    print(f"Validation data shape: {data['X_val_first'].shape}")
    print(f"Test data second shape: {data['X_test_second'].shape}")
    print(f"Validation data second shape: {data['X_val_second'].shape}")