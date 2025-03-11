
import numpy as np
import pyro
import torch
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




# Set random seed for reproducibility
SEED = 42
pyro.set_rng_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


# 1. Data Loading and Preprocessing
def load_and_preprocess_data():
    # Load Wine Quality dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')

    # Split features and target
    X = data.drop('quality', axis=1).values
    y = data['quality'].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED
    )

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    return X_train, y_train, X_test, y_test, data.columns[:-1]