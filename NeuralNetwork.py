import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork(nn.Module):
    def __init__(self, layers: List[int], epochs: int, batch_size: int, lr=0.001, weight_decay=0.001, dropout=0.3):
        super().__init__()
        self.layers = []
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size

        for i in range(1, len(layers)-1):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))
            self.layers.append(nn.ReLU())
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)

    def fit(self, x_train, y_train, x_val=None, y_val=None, patience=20):
        """
        Mimics sklearn fit function. trains the neural network on training data.
        :param x_train: np.array
        :param y_train: np.array
        :return: returns self
        """

        # initialize criterion for non-binary classification
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # PyToch needs data as tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        if x_val is not None and y_val is not None:
            x_val = torch.tensor(x_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.long)

        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=self.batch_size, shuffle=True)

        # Tackers for best epoch
        best_val_loss = float('inf')
        best_weights = None
        epochs_without_improvement = 0


        for epoch in range(self.epochs):
            self.train()
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self(x_batch), y_batch)
                loss.backward()
                optimizer.step()

            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = criterion(self(x_val), y_val)
                    val_acc = (self(x_val).argmax(dim=1) == y_val).float().mean()
                    print(f"Epoch {epoch}: Train loss: {loss.item():.4f}, Val loss: {val_loss.item():.4f}, Val acc: {val_acc:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = copy.deepcopy(self.state_dict())
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        if best_weights is not None:
            self.load_state_dict(best_weights)
        return self

    def predict_proba(self, data):
        """
        Mimics sklearn. Returns class probabilities.
        :param data: np.array
        :return:
        """
        self.eval()
        X_t = torch.tensor(np.asarray(data), dtype=torch.float32)
        with torch.no_grad():
            logits = self(X_t)

            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, data):
        """
        Mimics sklearn. predicted class values.
        :param data: np.array
        :return: np.array
        """
        probs = self.predict_proba(data)
        return np.argmax(probs, axis=1)