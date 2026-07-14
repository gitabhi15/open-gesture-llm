import socket
import numpy as np
import torch
import torch.nn as nn
import time

UDP_IP = "0.0.0.0"
UDP_PORT = 5005
SAMPLE_DURATION = 2  

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0)

# --- THE FIXED MODEL ARCHITECTURE ---
class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1472, num_classes) 
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
    print("✅ Model loaded successfully!")
    # This dynamically prints out whatever gestures you trained (e.g., G1, G2, G3, G4)
    print(f"🧠 Trained Gestures Detected: {', '.join(le.classes_)}") 
    print("-" * 30)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def flush_udp_buffer():
    sock.setblocking(False)
    while True:
        try: sock.recv(1024)
        except BlockingIOError: break
    sock.setblocking(True)
    sock.settimeout(1.0)

print("\n--- Push-to-Gesture Demo ---")

while True:
    cmd = input("\nPress ENTER to start gesture (or 'q' to quit): ")
    if cmd.lower() == 'q': break

    print("Listening... DO YOUR GESTURE NOW!")
    data = []
    
    flush_udp_buffer()
    start = time.time()

    while time.time() - start < SAMPLE_DURATION:
        try:
            packet, addr = sock.recvfrom(1024)
            line = packet.decode('utf-8').strip()
            if not line: continue
            ax, ay, az = map(float, line.split(','))
            data.append([ax, ay, az])
        except:
            pass

    data_arr = np.array(data)

    if len(data_arr) == 0:
        print("Error: No data received.")
        continue

    if len(data_arr) < 50:
        last_value = data_arr[-1:]
        pad = np.repeat(last_value, 50 - len(data_arr), axis=0)
        data_arr = np.vstack([data_arr, pad])
    else:
        data_arr = data_arr[:50]

    data_tensor = data_arr.T  
    data_tensor = torch.tensor(data_tensor).unsqueeze(0).float()

    with torch.no_grad():
        out = model(data_tensor)
        probs = torch.nn.functional.softmax(out, dim=1).numpy()[0]
        
        # This will automatically list confidence percentages for all trained gestures
        confidence_dict = {le.inverse_transform([i])[0]: f"{p*100:.1f}%" for i, p in enumerate(probs)}
        
        pred = torch.argmax(out).item()
        gesture = le.inverse_transform([pred])[0]

    print(f"\n✅ PREDICTION: {gesture}")
    print(f"   Confidence Breakdown: {confidence_dict}")
    print("-" * 30)