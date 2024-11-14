
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.utils import resample


def load_spotify_dataset():
    """
    Load the Spotify dataset from a CSV file, preprocess it, and standardize the numeric columns.

    The dataset is expected to be located at '../data/raw/dataset.csv'.
    The first column of the CSV file is used as the index.

    The function performs the following preprocessing steps:
    1. Drops duplicate rows.
    2. Handles missing values by dropping rows with any missing values.
    3. Upsamples the minority class in 'explicit' to balance it.
    4. Balances data across 'time_signature' by upsampling to match the maximum count.
    5. One-hot encodes the 'explicit' column.
    6. Label encodes other categorical columns.
    7. Standardizes the numeric columns to have mean 0 and standard deviation 1.
    8. Separates features (X) and target (y) after all preprocessing steps.

    Returns:
        pd.DataFrame: Preprocessed features (X).
        pd.Series: Target variable 'track_genre' (y).
    """
    df_spotify = pd.read_csv('../data/raw/dataset.csv', index_col=0)

    # Drop duplicate rows
    df_spotify = df_spotify.drop_duplicates()

    # Handle missing values
    df_spotify = df_spotify.dropna()

    # Upsample the minority class in 'explicit' to balance it
    df_majority = df_spotify[df_spotify['explicit'] == False]
    df_minority = df_spotify[df_spotify['explicit'] == True]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )
    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    # Balance across 'time_signature' by upsampling
    df_resampled_time_signature = df_balanced.groupby('time_signature').apply(
        lambda x: x.sample(df_balanced['time_signature'].value_counts().max(), replace=True)
    ).reset_index(drop=True)

    # One-Hot Encode the 'explicit' column
    df_resampled_time_signature = pd.get_dummies(df_resampled_time_signature, columns=['explicit'], prefix='explicit')

    # Label encode other categorical columns
    categorical_cols = df_resampled_time_signature.select_dtypes(include=['object']).columns.difference(['track_genre'])
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_resampled_time_signature[col] = le.fit_transform(df_resampled_time_signature[col])
        label_encoders[col] = le

    # Standardize numeric columns
    df_resampled_time_signature = standardize_data(df_resampled_time_signature)

    # Separate X and y after all preprocessing
    X = df_resampled_time_signature.drop(columns=['track_genre'])
    y = df_resampled_time_signature['track_genre']

    return X, y

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

    scaler = StandardScaler()
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
