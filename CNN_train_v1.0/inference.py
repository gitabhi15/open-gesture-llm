import serial
import numpy as np
import torch
import torch.nn as nn
import time
import joblib

PORT = '/dev/ttyUSB0'
BAUD = 115200

ser = serial.Serial(PORT, BAUD)

# Model (same as training)
class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Load model
checkpoint = torch.load("model.pth")
le = checkpoint["label_encoder"]

model = GestureCNN(len(le.classes_))
model.load_state_dict(checkpoint["model_state"])
model.eval()

buffer = []

def predict(buffer):
    data = np.array(buffer)

    if len(data) < 50:
        return None

    data = data[:50]
    data = data.T
    data = torch.tensor(data).unsqueeze(0).float()

    out = model(data)
    pred = torch.argmax(out).item()
    return le.inverse_transform([pred])[0]

print("Running real-time gesture recognition...")

while True:
    try:
        line = ser.readline().decode().strip()
        ax, ay, az = map(float, line.split(','))

        buffer.append([ax, ay, az])

        if len(buffer) > 50:
            buffer.pop(0)

        gesture = predict(buffer)

        if gesture:
            print("Detected:", gesture)

    except:
        pass