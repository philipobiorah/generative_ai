import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer: 784 neurons, Hidden layer: 128 neurons
        self.fc2 = nn.Linear(128, 10)   # Hidden layer: 128 neurons, Output layer: 10 neurons

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MyMLP(nn.Module):
    """My Multilayer Perceptron (MLP)

    Specifications:

        - Input layer: 784 neurons
        - Hidden layer: 128 neurons with ReLU activation
        - Output layer: 10 neurons with softmax activation

    """

    def __init__(self):
        super(MyMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer: 784 neurons, Hidden layer: 128 neurons
        self.fc2 = nn.Linear(128, 10)   # Hidden layer: 128 neurons, Output layer: 10 neurons
        self.relu = nn.ReLU()           # ReLU activation for hidden layer
        self.softmax = nn.Softmax(dim=1) # Softmax activation for output layer

    def forward(self, x):
        # Pass the input to the first layer
        x = self.fc1(x)

        # Apply ReLU activation
        x = self.relu(x)

        # Pass the result to the final layer
        x = self.fc2(x)

        # Apply softmax activation
        x = self.softmax(x)
        
        return x

def fake_training_loaders():
    for _ in range(30):
        yield torch.randn(64, 784), torch.randint(0, 10, (64,))

# Example usage
if __name__ == "__main__":
    model = MyMLP()
    print(model)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer (by convention we use the variable optimizer)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(3):
        # Create a training loop
        for i, data in enumerate(fake_training_loaders()):
            # Every data instance is an input + label pair
            x, y = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Forward pass (predictions)
            y_pred = model(x)

            # Compute the loss and its gradients
            loss = loss_fn(y_pred, y)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")