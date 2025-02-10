# neural_network.py

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from data.data_loader import load_spotify_dataset, train_val_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_neural_network():
    logging.info("Loading and preprocessing data...")
    X, y = load_spotify_dataset()
    X_train, y_train, X_val, y_val, _, _ = train_val_test_split(X, y)
    logging.info("Data loaded and split into training and validation sets.")

    # Convert boolean columns to int
    bool_cols = X_train.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        logging.info(f"Boolean columns detected: {bool_cols.tolist()}")
        X_train[bool_cols] = X_train[bool_cols].astype(int)
        X_val[bool_cols] = X_val[bool_cols].astype(int)

    # Convert 'True'/'False' strings to 1 and 0
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            unique_values = X_train[col].unique()
            if set(unique_values).issubset({'True', 'False'}):
                logging.info(f"Converting column {col} from 'True'/'False' to 1/0.")
                X_train[col] = X_train[col].map({'True': 1, 'False': 0}).astype(int)
                X_val[col] = X_val[col].map({'True': 1, 'False': 0}).astype(int)

    # Identify non-numeric columns
    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        logging.info(f"Non-numeric columns detected: {non_numeric_cols}")
        # Attempt to convert to numeric
        for col in non_numeric_cols:
            try:
                X_train[col] = pd.to_numeric(X_train[col])
                X_val[col] = pd.to_numeric(X_val[col])
            except ValueError:
                logging.warning(f"Column {col} cannot be converted to numeric, dropping it.")
                X_train.drop(columns=[col], inplace=True)
                X_val.drop(columns=[col], inplace=True)
    else:
        logging.info("No non-numeric columns detected.")

    # Align columns between X_train and X_val
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    # Ensure all columns are numeric
    try:
        X_train = X_train.apply(pd.to_numeric)
        X_val = X_val.apply(pd.to_numeric)
    except Exception as e:
        logging.error(f"Error converting data to numeric: {e}")
        raise

    # Handle any NaN values that may have been introduced
    X_train.fillna(0, inplace=True)
    X_val.fillna(0, inplace=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the model
    input_size = X_train.shape[1]
    num_classes = len(y_train.unique())
    model = NeuralNetwork(input_size, num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 20
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_accuracy = correct / total
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Return the model and validation accuracy
    trained_model = NeuralNetworkModel(model, device, X_train.columns)
    return trained_model, val_accuracy

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

class NeuralNetworkModel:
    def __init__(self, model, device, feature_columns):
        self.model = model
        self.device = device
        self.feature_columns = feature_columns

    def predict(self, X):
        """
        Predict the labels for the given data X.

        Args:
            X: Features to predict on.

        Returns:
            y_pred: Predicted labels.
        """
        # Ensure data is numeric
        # Convert boolean columns to int
        bool_cols = X.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            X[bool_cols] = X[bool_cols].astype(int)

        # Convert 'True'/'False' strings to 1 and 0
        for col in X.columns:
            if X[col].dtype == 'object':
                if set(X[col].unique()).issubset({'True', 'False'}):
                    X[col] = X[col].map({'True': 1, 'False': 0}).astype(int)

        # Identify non-numeric columns
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            for col in non_numeric_cols:
                try:
                    X[col] = pd.to_numeric(X[col])
                except ValueError:
                    X.drop(columns=[col], inplace=True)

        # Align columns to training data
        X = X.reindex(columns=self.feature_columns, fill_value=0)

        # Handle any NaN values that may have been introduced
        X.fillna(0, inplace=True)

        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()
