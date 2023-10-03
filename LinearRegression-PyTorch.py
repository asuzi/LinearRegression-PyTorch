import torch
from torch import nn
import matplotlib.pyplot as plt
import os

LR = 0.01
EPOCH = 1000
MODEL_PATH = f"{os.getcwd()}\\linear_regression_model_exercise.pth"


def main():
    TRAINING = True
    TESTING = True

    if TRAINING:
        train()

    if TESTING:
        test()


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

w, b = 0.3, 0.7
start, end, step = 0, 1, 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1).to(device)

y = w * X + b

split = int(0.8 * len(X))
train_X = X[:split]
test_X = X[split:]

split = int(0.8 * len(y))
train_y = X[:split]
test_y = X[split:]

print(f"Splitting X, Training data: {train_X.shape}. Testing data: {test_X.shape}.")
print(f"Splitting Y, Training data: {train_y.shape}. Testing data: {test_y.shape}.")


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def train():
    best = 9999

    model = LinearRegression()
    model.train()
    model.to(device)

    loss_fn = nn.L1Loss()
    optim = torch.optim.SGD(model.parameters(), LR)

    for _ in range(EPOCH):
        predictions = model(train_X)

        loss = loss_fn(predictions, train_y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if loss < best:
            best = loss
            torch.save(model.state_dict(), MODEL_PATH)
    print("Best training loss: ", best)


def test():
    model = LinearRegression()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    model.to(device)

    loss_fn = nn.L1Loss()

    with torch.inference_mode():
        predictions = model(test_X)
        loss = loss_fn(predictions, test_y)

    print("Testing loss: ", loss)
    plot_predictions(
        train_X.cpu(), train_y.cpu(), test_X.cpu(), test_y.cpu(), predictions.cpu()
    )


# Helper for plotting the data
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    Plots training data, test data and compares predictions
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions != None:
        # Plot prediction if exists
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


if __name__ == "__main__":
    main()
