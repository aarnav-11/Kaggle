import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#we will train a binary classifier (NN)

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

train = pd.get_dummies(train, columns=["Embarked"], dtype=int)
train["HasCabin"] = train["Cabin"].notna().astype(int)
train["NumCabins"] = train["Cabin"].fillna("").apply(
    lambda x: len(x.split()) if x else 0
)
train["Deck"] = train["Cabin"].str[0].fillna("Unkown")
train = pd.get_dummies(train, columns=["Deck"], dtype=int)
train = pd.get_dummies(train, columns=["Sex"], dtype=int)
train["Age"] = train["Age"].fillna(train["Age"].median())

Y = train["Survived"]
train = train.drop(columns=["Name", "Cabin", "Survived", "PassengerId", "Ticket"])
X = train


X = torch.tensor(train.values, dtype=torch.float32)
Y = torch.tensor(Y.values, dtype=torch.float32).reshape(-1, 1)
# print(X.shape)

# print(X.head())
# print(Y.head())

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
model = NeuralNetwork(X.shape[1])

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


for epoch in range(1000):
    model.train()
    logits = model(X)
    loss = loss_fn(logits, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
