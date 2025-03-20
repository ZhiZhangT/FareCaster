# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


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


def preprocess_dataframe(df, fit=True, le_start=None, le_dest=None):
    """
    Preprocess the DataFrame by parsing dates and encoding categorical features.
    Scaling is postponed so that we can keep the DataFrame for splitting and sorting.
    """
    df = df.copy()

    # Parse dates and convert to datetime objects (assuming parse_date is defined)
    df["searchDate"] = df["searchDate"].apply(parse_date)
    df["flightDate"] = df["flightDate"].apply(parse_date)

    # Create ordinal columns (useful for modeling)
    df["searchDate_ordinal"] = df["searchDate"].apply(lambda x: x.toordinal())
    df["flightDate_ordinal"] = df["flightDate"].apply(lambda x: x.toordinal())

    # Encode categorical features
    if fit:
        le_start = LabelEncoder()
        le_dest = LabelEncoder()
        df["startingAirport_enc"] = le_start.fit_transform(df["startingAirport"])
        df["destinationAirport_enc"] = le_dest.fit_transform(df["destinationAirport"])
    else:
        if le_start is None or le_dest is None:
            raise ValueError(
                "Pre-fitted label encoders must be provided when fit=False."
            )
        df["startingAirport_enc"] = le_start.transform(df["startingAirport"])
        df["destinationAirport_enc"] = le_dest.transform(df["destinationAirport"])

    return df, (le_start, le_dest)


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
