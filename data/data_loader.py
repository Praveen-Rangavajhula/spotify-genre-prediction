import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample


def load_spotify_dataset(
    input_path='../data/raw/dataset.csv',
    save_preprocessed=False,
    preprocessed_path='../data/processed/dataset_preprocessed.csv'
):
    """
    Load and preprocess the Spotify dataset.

    This function performs the following steps:
    - Load the dataset from a CSV file.
    - Drop duplicates and handle missing values.
    - Balance the 'explicit' feature via upsampling.
    - Balance the 'time_signature' feature via upsampling.
    - One-hot encode categorical features ('explicit', 'key', 'mode').
    - Label encode other categorical features except 'track_genre'.
    - Create interaction features.
    - Remove highly correlated features.
    - Transform skewed features using log transformation.
    - Standardize numeric columns.
    - Label encode the target variable 'track_genre'.
    - Separate features and target variable.
    - Optionally save the preprocessed dataset to a CSV file.

    Args:
        input_path (str): Path to the raw dataset CSV file.
        save_preprocessed (bool): If True, saves the preprocessed dataset to a CSV file.
        preprocessed_path (str): Path to save the preprocessed dataset.

    Returns:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
    """
    # Load the dataset
    df_spotify = pd.read_csv(input_path, index_col=0)

    # Drop duplicate rows and handle missing values
    df_spotify = df_spotify.drop_duplicates().dropna()

    # Upsample the minority class in 'explicit' to balance it
    df_majority = df_spotify[df_spotify['explicit'] == False]
    df_minority = df_spotify[df_spotify['explicit'] == True]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,             # Sample with replacement
        n_samples=len(df_majority),  # Match number in majority class
        random_state=42           # Reproducible results
    )
    df_balanced_explicit = pd.concat([df_majority, df_minority_upsampled])

    # Balance the 'time_signature' feature via upsampling
    max_size = df_balanced_explicit['time_signature'].value_counts().max()
    df_balanced_time_signature = df_balanced_explicit.groupby('time_signature').apply(
        lambda x: x.sample(max_size, replace=True, random_state=42)
    ).reset_index(drop=True)

    # One-hot encode 'explicit', 'key', and 'mode' columns
    df_encoded = pd.get_dummies(
        df_balanced_time_signature,
        columns=['explicit', 'key', 'mode'],
        prefix=['explicit', 'key', 'mode'],
        drop_first=True
    )

    # Label encode other categorical columns except 'track_genre'
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.difference(['track_genre'])
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Create interaction features
    df_encoded['danceability_valence'] = (
        df_encoded['danceability'] * df_encoded['valence']
    )
    df_encoded['energy_acousticness'] = (
        df_encoded['energy'] * df_encoded['acousticness']
    )

    # Remove highly correlated feature 'loudness'
    df_encoded = df_encoded.drop(columns=['loudness'])

    # Transform skewed features using log1p
    skewed_features = ['instrumentalness', 'speechiness', 'liveness']
    for feature in skewed_features:
        df_encoded[feature] = np.log1p(df_encoded[feature])

    # Standardize numeric columns
    df_encoded = standardize_data(df_encoded)

    # Label encode the target variable 'track_genre'
    le_genre = LabelEncoder()
    df_encoded['track_genre'] = le_genre.fit_transform(df_encoded['track_genre'])

    # Separate features and target variable
    X = df_encoded.drop(columns=['track_genre'])
    y = df_encoded['track_genre']

    # Optionally save the preprocessed dataset
    if save_preprocessed:
        df_encoded.to_csv(preprocessed_path, index=False)
        print(f"Preprocessed dataset saved to {preprocessed_path}")

    return X, y


def standardize_data(df):
    """
    Standardize the numeric columns of the DataFrame using StandardScaler.

    This function scales the numeric columns to have mean 0 and standard deviation 1.

    Args:
        df (pd.DataFrame): The input DataFrame to be standardized.

    Returns:
        pd.DataFrame: The DataFrame with standardized numeric columns.
    """
    # Exclude dummy variables from scaling
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def train_val_test_split(X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        train_size (float): Proportion of the dataset to include in the train split.
        val_size (float): Proportion of the dataset to include in the validation split.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator.

    Returns:
        X_train, y_train: Training set.
        X_val, y_val: Validation set.
        X_test, y_test: Test set.
    """
    assert train_size + val_size + test_size == 1, "train, val, and test sizes must sum to 1"

    # Split the dataset with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )

    # Calculate the validation ratio
    val_ratio = val_size / (val_size + test_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, random_state=random_state, stratify=y_temp
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
