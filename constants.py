SAVED_DIR = "FareCaster/saved/"
RANDOM_STATE = 42
VALIDATION_FILEPATH = "FareCaster/data/itineraries_10perc_validation.csv"
TEST_FILEPATH = "data/itineraries_10perc_test.csv"
RAW_DATA_PATH = "FareCaster/data/itineraries_10perc.csv"
PROCESSED_DATA_FILE = f"{SAVED_DIR}preprocessed_data.pkl"
PROCESSED_DATA_FILE_SOTA = f"{SAVED_DIR}preprocessed_data_sota.pkl"
SCALER_FILE = f"{SAVED_DIR}scaler.pkl"
TARGET_COL = "totalFare"
FEATURE_COLS = [
    # "startingAirport_enc",
    # "destinationAirport_enc",
    # "seatsRemaining",
    # "searchDayOfWeek",
    # "flightDayOfWeek",
    # "daysBetweenSearchAndFlight",
    # "searchMonth",
    # "flightMonth",
    # "departureHourUTC",
    TARGET_COL,
]
CATEGORICAL_COLS = [
    "searchDayOfWeek",
    "flightDayOfWeek",
    "startingAirport_enc",
    "destinationAirport_enc",
    "searchMonth",
    "flightMonth",
    "departureHourUTC",
]
CYCLICAL_FEATURES = {
    "searchDayOfWeek": 7,
    "flightDayOfWeek": 7,
    "searchMonth": 12,
    "flightMonth": 12,
    "departureHourUTC": 24,
}
