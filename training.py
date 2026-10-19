import copy
from enum import Enum
from typing import Tuple, List, Dict, Optional, Any, Callable

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid

from NeuralNetwork import NeuralNetwork

DATA_DIR = "./task1all6runs"


class FeatureTypes(Enum):
    RAW = 1
    PCA = 2
    SCALED = 3


def load_splits(run_number: int, feature_type: FeatureTypes) -> Tuple:
    """
    Returns the training/validation/test data from the generated files.
    :param run_number: run number (1 to 6)
    :param feature_type: FeatureTypes enum (RAW, PCA, SCALED)
    :return: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    file_path = f"{DATA_DIR}/run_{run_number}_{feature_type.name.lower()}_features.npz"
    with np.load(file_path) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]

        X_val = data["X_validation"]
        y_val = data["y_validation"]

        X_test = data["X_test"]
        y_test = data["y_test"]
    if feature_type == FeatureTypes.PCA:
        X_train = X_train[:, :38]
        X_val = X_val[:, :38]
        X_test = X_test[:, :38]
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_metrics(model, X_test, y_test, labels: Optional[np.ndarray] = None):
    y_pred = model.predict(X_test)

    acc_score = metrics.accuracy_score(y_test, y_pred)
    balanced_acc_score = metrics.balanced_accuracy_score(y_test, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=labels)
    macro_f1_score = metrics.f1_score(y_test, y_pred, average="macro")

    return acc_score, confusion_matrix, balanced_acc_score, macro_f1_score


def _clone_model(model):
    """
    Creates a fresh, untrained copy of the given model.
    Supports PyTorch nn.Module, scikit-learn estimators, or callable factories.
    """
    if isinstance(model, torch.nn.Module):
        model_copy = copy.deepcopy(model)
        for module in model_copy.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        return model_copy
    elif callable(model) and not hasattr(model, "fit"):
        return model()
    else:
        try:
            return clone(model)
        except Exception:
            return copy.deepcopy(model)

def real_k_fold(
    model,
    feature_type: FeatureTypes,
):
    confusion_matrices = []
    acc_val_scores = []
    acc_test_scores = []
    balanced_acc_test_scores = []
    macro_f1_test_scores = []
    for i in range(6):
        X_train, y_train, X_val, y_val, X_test, y_test = load_splits(i+1, feature_type)
        fold_model = _clone_model(model)
        if isinstance(fold_model, torch.nn.Module):
            fold_model.fit(X_train, y_train, X_val, y_val)
        else:
            fold_model.fit(X_train, y_train)
        acc_val, _, _, _ = get_metrics(fold_model, X_val, y_val)
        acc_test, cm_test, balanced_acc_test, macro_f1_test = get_metrics(fold_model, X_test, y_test)
        acc_val_scores.append(acc_val)
        acc_test_scores.append(acc_test)
        balanced_acc_test_scores.append(balanced_acc_test)
        confusion_matrices.append(cm_test)
        macro_f1_test_scores.append(macro_f1_test)

    mean_acc_val = np.mean(acc_val_scores)
    mean_acc_test = np.mean(acc_test_scores)
    mean_balanced_acc_test = np.mean(balanced_acc_test_scores)
    mean_macro_f1_test = np.mean(macro_f1_test_scores)
    std_acc_val = np.std(acc_val_scores)
    std_acc_test = np.std(acc_test_scores)
    std_balanced_acc_test = np.std(balanced_acc_test_scores)
    std_macro_f1_test = np.std(macro_f1_test_scores)
    total_cm = np.sum(confusion_matrices, axis=0)

    return {
        "mean_acc_val": mean_acc_val,
        "mean_acc_test": mean_acc_test,
        "mean_balanced_acc_test": mean_balanced_acc_test,
        "mean_macro_f1_test": mean_macro_f1_test,
        "std_acc_val": std_acc_val,
        "std_acc_test": std_acc_test,
        "std_balanced_acc_test": std_balanced_acc_test,
        "std_macro_f1_test": std_macro_f1_test,
        "confusion_matrices": confusion_matrices,
        "total_cm": total_cm,
    }


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(6, FeatureTypes.PCA)
    print(
        f"Original Split -> Training: {X_train.shape} | "
        f"Validation: {X_val.shape} | "
        f"Test: {X_test.shape}")

    # Use first 4 principal components
    X_train = X_train[:, :4]
    X_val = X_val[:, :4]
    X_test = X_test[:, :4]

    # Combine train and validation sets for k-fold cross-validation instead of naive validation
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    print(f"Combined Training Set (Train + Val) for K-Fold CV: {X_train_full.shape}")

    input_dim = X_train_full.shape[1]

    # 1. Neural Network with K-Fold Cross-Validation
    print("\n================== Neural Network ==================")
    neural = NeuralNetwork([input_dim, 16, 8, 4], epochs=100, batch_size=16)
    nn_cv_results = k_fold_cross_validation(
        neural, X_train_full, y_train_full, k=5, stratified=True, random_state=42
    )

    # Train final neural network on the full training set and evaluate on unseen test set
    final_neural = _clone_model(neural)
    final_neural.fit(X_train_full, y_train_full)
    nn_test_acc, nn_test_cm = get_metrics(final_neural, X_test, y_test)
    print(f"Neural Network Test Accuracy: {nn_test_acc:.4f}")
    print(f"Neural Network Test Confusion Matrix:\n{nn_test_cm}")

    # 2. Random Forest with K-Fold Cross-Validation
    print("\n================= Random Forest ===================")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_cv_results = k_fold_cross_validation(
        rf, X_train_full, y_train_full, k=5, stratified=True, random_state=42
    )
    final_rf = clone(rf)
    final_rf.fit(X_train_full, y_train_full)
    rf_test_acc, rf_test_cm = get_metrics(final_rf, X_test, y_test)
    print(f"Random Forest Test Accuracy: {rf_test_acc:.4f}")
    print(f"Random Forest Test Confusion Matrix:\n{rf_test_cm}")
