from typing import List

import torch.nn as nn
import torch.optim as optim

learning_rate = 0.001


class NeuralNetwork(nn.Module):
    def __init__(self, layers: List[int]):
        super().__init__()
        self.activation = nn.ReLU()
        self.layers = []
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.outputLayer(x)
        return x

    def train_model(self, x_train, y_train, epochs):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            prediction = self.forward(x_train)
            loss = criterion(prediction, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

