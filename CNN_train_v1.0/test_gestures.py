import socket
import numpy as np
import torch
import torch.nn as nn
import time

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(2.0)

# --- MODEL DEFINITION ---
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

# --- LOAD MODEL ---
try:
    checkpoint = torch.load("model.pth", weights_only=False)
    le = checkpoint["label_encoder"]
    model = GestureCNN(len(le.classes_))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
except Exception as e:
    print(f"Could not load model.pth. Did you run train_model.py first? Error: {e}")
    exit()

# --- PIPELINE STATE ---
state = "IDLE"
short_buffer = []       
recording_buffer = []   
cooldown_start = 0

COOLDOWN = 1.0          
WINDOW_SIZE = 50        
TRIGGER_SIZE = 10       
ACTIVITY_THRESHOLD = 0.3 

print(f"System Ready! Listening on Wi-Fi Port {UDP_PORT} for movement...")

while True:
    try:
        packet, addr = sock.recvfrom(1024)
        line = packet.decode('utf-8').strip()
        if not line: continue
        
        ax, ay, az = map(float, line.split(','))

        if state == "IDLE":
            short_buffer.append([ax, ay, az])
            if len(short_buffer) > TRIGGER_SIZE:
                short_buffer.pop(0)

            if len(short_buffer) == TRIGGER_SIZE:
                std_devs = np.std(short_buffer, axis=0)
                activity_level = np.sum(std_devs)

                if activity_level > ACTIVITY_THRESHOLD:
                    print(f"--> Motion Detected! (Activity: {activity_level:.2f}) Recording...")
                    state = "RECORDING"
                    recording_buffer = list(short_buffer)

        elif state == "RECORDING":
            recording_buffer.append([ax, ay, az])

            if len(recording_buffer) == WINDOW_SIZE:
                data_arr = np.array(recording_buffer)
                data_tensor = data_arr.T  
                data_tensor = torch.tensor(data_tensor).unsqueeze(0).float()

                out = model(data_tensor)
                pred = torch.argmax(out).item()
                gesture = le.inverse_transform([pred])[0]

                print(f"\n[SUCCESS] Detected Gesture: {gesture}")
                print("-" * 30)
                
                state = "COOLDOWN"
                cooldown_start = time.time()
                short_buffer.clear()

        elif state == "COOLDOWN":
            if time.time() - cooldown_start > COOLDOWN:
                state = "IDLE"

    except socket.timeout:
        # Ignore timeouts, just means no data is arriving
        pass
    except Exception as e:
        # Ignore malformed packets dropping
        pass