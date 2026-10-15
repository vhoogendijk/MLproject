from enum import Enum
from typing import Tuple

import torch
from sklearn.metrics import accuracy_score

from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

DATA_DIR = "./task1all6runs"

class FeatureTypes(Enum):
    RAW = 1
    PCA = 2
    SCALED = 3

def load_splits(run_number, feature_type: FeatureTypes) -> Tuple:
    """
    Returns de training/test data from the generated files
    :param run_number:
    :param feature_type:
    :return:
    """
    file_path = f"{DATA_DIR}/run_{run_number}_{feature_type.name.lower()}_features.npz"
    with np.load(file_path) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]

        X_val = data["X_validation"]
        y_val = data["y_validation"]

        X_test = data["X_test"]
        y_test = data["y_test"]
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc_score = metrics.accuracy_score(y_test, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    return acc_score, confusion_matrix

def tune_hyperparameters(model, X_train, y_train, X_test, y_test):


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(6, FeatureTypes.PCA)
    print(
        f"Training set size: {X_train.shape} | "
        f"Validation set size: {X_val.shape} | "
        f"Test set size: {X_test.shape}")
    X_train = X_train[:, :4]
    X_val = X_val[:, :4]
    X_test = X_test[:, :4]

    # Optional: If you want to combine train and validation sets instead of tracking val_acc:
    # X_train = np.concatenate([X_train, X_val], axis=0)
    # y_train = np.concatenate([y_train, y_val], axis=0)

    input_dim = X_train.shape[1]  # 94 for SCALED, automatically adapts if you switch to PCA

    neural = NeuralNetwork([input_dim, 16, 8, 4], epochs=100, batch_size=16)
    neural.fit(X_train, y_train, x_val=X_val, y_val=y_val)

    print(f"accuracy + CM: {get_metrics(neural, X_test, y_test)}")
