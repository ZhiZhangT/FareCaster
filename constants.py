SAVED_DIR = "saved/"
RANDOM_STATE = 42
VALIDATION_FILEPATH = "data/itineraries_10perc_validation.csv"
TEST_FILEPATH = "data/itineraries_10perc_test.csv"
RAW_DATA_PATH = "data/itineraries_10perc.csv"
PROCESSED_DATA_FILE = f"{SAVED_DIR}preprocessed_data.pkl"
SCALER_FILE = f"{SAVED_DIR}scaler.pkl"
FEATURE_COLS = [
    "searchDate_ordinal",
    "flightDate_ordinal",
    "startingAirport_enc",
    "destinationAirport_enc",
    "seatsRemaining",
    "searchDayOfWeek",
    "flightDayOfWeek",
    "daysBetweenSearchAndFlight",
    "searchMonth",
    "flightMonth",
    "departureHourUTC",
]
