
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_spotify_dataset():
    """
    Load the Spotify dataset from a CSV file, preprocess it, and standardize the numeric columns.

    The dataset is expected to be located at '../data/raw/dataset.csv'.
    The first column of the CSV file is used as the index.

    The function performs the following preprocessing steps:
    1. Drops duplicate rows.
    2. Handles missing values by dropping rows with any missing values.
    3. Extracts the 'track_genre' column as the target variable.
    4. Drops the 'track_genre' and 'track_id' columns from the dataset.
    5. One-hot encodes the 'explicit' column.
    6. Standardizes the numeric columns to a range of [0, 1].

    Returns:
        pd.DataFrame: A DataFrame containing the preprocessed and standardized Spotify dataset.
        pd.Series: A Series containing the target variable 'track_genre'.
    """
    df_spotify = pd.read_csv('../data/raw/dataset.csv', index_col=0)

    # Drop duplicate rows
    df_spotify = df_spotify.drop_duplicates()

    # Handle missing values
    df_spotify = df_spotify.dropna()

    target_column = df_spotify['track_genre']

    # Drop target column and track_id
    df_spotify = df_spotify.drop(columns=['track_genre', 'track_id'])

    df_spotify = pd.get_dummies(df_spotify, columns=['explicit'], prefix='explicit')

    df_spotify = standardize_data(df_spotify)

    return df_spotify, target_column


def standardize_data(df):
    """
    Standardize the numeric columns of the DataFrame using MinMaxScaler.

    The function scales the numeric columns of the DataFrame to a range of [0, 1].

    Args:
        df (pd.DataFrame): The input DataFrame to be standardized.

    Returns:
        pd.DataFrame: The DataFrame with standardized numeric columns.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def train_val_test_split(X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """
    Splits the features and target into train, validation, and test sets.

    Args:
        X (pd.DataFrame): The input DataFrame containing features.
        y (pd.Series): The target Series.
        train_size (float): Proportion of data to use for the training set.
        val_size (float): Proportion of data to use for the validation set.
        test_size (float): Proportion of data to use for the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Training features
        pd.Series: Training target
        pd.DataFrame: Validation features
        pd.Series: Validation target
        pd.DataFrame: Test features
        pd.Series: Test target
    """
    assert train_size + val_size + test_size == 1, "train, val, and test sizes must sum to 1"

    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_size, random_state=random_state)

    val_ratio = val_size / (val_size + test_size)

    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=1 - val_ratio, random_state=random_state)

    return X_train, y_train, X_val, y_val, X_test, y_test
