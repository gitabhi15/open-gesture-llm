import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {DEVICE}")

EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_CLASSES = 4

class SensorDataset(Dataset):
    def __init__(self, x_path, y_path):
        X_numpy = np.load(x_path)
        y_numpy = np.load(y_path)

        X_numpy = np.transpose(X_numpy, (0, 2, 1))

        self.X = torch.tensor(X_numpy, dtype=torch.float32)
        self.y = torch.tensor(y_numpy, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GestureNet(nn.Module):
    def __init__(self, num_classes):
        super(GestureNet, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.conv_block(x)
        output = self.classifier(features)
        return output


def main():
    dataset = SensorDataset("training_tensors_data/X_train.npy", "training_tensors_data/y_train.npy")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GestureNet(NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting Training Loop...")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct_predictions = 0

        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(batch_X)

            loss = criterion(predictions, batch_y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            _, predicted_classes = torch.max(predictions, 1)
            correct_predictions += (predicted_classes == batch_y).sum().item()

        epoch_accuracy = (correct_predictions / len(dataset)) * 100
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")

    torch.save(model.state_dict(), "gesture_model.pth")
    print("\nTraining Complete! Model saved as 'gesture_model.pth'")

if __name__ == "__main__":
    main()