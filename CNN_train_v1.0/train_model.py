import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Load data
X = []
y = []

file_list = glob.glob("data/*.npy")
if not file_list:
    print("No .npy files found in data/ directory!")
    exit()

for file in file_list:
    data = np.load(file)

    if len(data) < 50:
        last_value = data[-1:]
        pad = np.repeat(last_value, 50 - len(data), axis=0)
        data = np.vstack([data, pad])
    else:
        data = data[:50]

    X.append(data)
    # Handle paths correctly whether on Windows or Linux
    filename = file.replace("\\", "/").split("/")[-1]
    label = filename.split("_")[0]
    y.append(label)

X = np.array(X)              # (batch, time, channels)
X = X.transpose(0, 2, 1)     # → (batch, channels, time)

print("Dataset distribution:", Counter(y))

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

# --- THE FIXED MODEL ARCHITECTURE ---
class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            # REMOVED AdaptiveAvgPool1d! We must preserve temporal sequence.
            nn.Flatten(),
            # 50 timesteps -> Conv(k=3) -> 48 -> Conv(k=3) -> 46. 
            # 46 timesteps * 32 channels = 1472 features.
            nn.Linear(1472, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = GestureCNN(len(set(y_encoded)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
print("Starting training...")
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    pred = torch.argmax(outputs, dim=1)
    acc = (pred == y_tensor).float().mean()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | Acc: {acc:.2f}")

# Save model
torch.save({
    "model_state": model.state_dict(),
    "label_encoder": le
}, "model.pth")

print("Model saved to model.pth!")