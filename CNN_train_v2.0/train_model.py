import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SensorDataset(Dataset):
    def __init__(self, x_path, y_path):
        X_numpy = np.transpose(np.load(x_path), (0, 2, 1))
        self.X = torch.tensor(X_numpy, dtype=torch.float32)
        self.y = torch.tensor(np.load(y_path), dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class GestureNet(nn.Module):
    def __init__(self, num_classes):
        super(GestureNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(10, 32, kernel_size=7, dilation=2, padding=6),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, dilation=2, padding=6),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.conv_block(x))

dataset = SensorDataset("training_tensors_data/X_train.npy", "training_tensors_data/y_train.npy")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

model = GestureNet(4).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(30):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_X.to(DEVICE)), batch_y.to(DEVICE))
        loss.backward()
        optimizer.step()
        
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            preds = model(batch_X.to(DEVICE))
            correct += (preds.argmax(1) == batch_y.to(DEVICE)).sum().item()
    print(f"Epoch {epoch+1} | Val Accuracy: {correct/val_size * 100:.1f}%")

torch.save(model.state_dict(), "gesture_model.pth")