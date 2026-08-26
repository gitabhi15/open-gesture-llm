import socket
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt

# ==========================================
# Configuration & Setup
# ==========================================
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
WINDOW_SIZE = 50
SAMPLING_RATE = 50.0
NUM_CLASSES = 4

# Map the output integers back to human-readable text
CLASS_MAP = {0: "Idle", 1: "Flick", 2: "Swivel", 3: "Wave"}

DEVICE = torch.device("cpu") # For live testing 50 rows, CPU is virtually instantaneous

# ==========================================
# DSP & Model Architecture (Must match exactly)
# ==========================================
def apply_filters(data_column, cutoff_low=0.5, cutoff_high=5.0):
    nyquist = 0.5 * SAMPLING_RATE
    # High-pass (remove gravity)
    b_high, a_high = butter(4, cutoff_low / nyquist, btype='high')
    no_gravity = filtfilt(b_high, a_high, data_column)
    # Low-pass (remove noise)
    b_low, a_low = butter(4, cutoff_high / nyquist, btype='low')
    return filtfilt(b_low, a_low, no_gravity)

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
        return self.classifier(self.conv_block(x))

# ==========================================
# Live Inference Loop
# ==========================================
def main():
    print("Loading GestureNet model...")
    model = GestureNet(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("gesture_model.pth", map_location=DEVICE))
    model.eval() # Set model to evaluation mode (disables training features)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    # The sliding window buffer
    window = deque(maxlen=WINDOW_SIZE)
    
    print(f"Listening for UDP stream on port {UDP_PORT}...")
    print("Perform a gesture!\n")
    
    try:
        while True:
            packet, _ = sock.recvfrom(1024)
            line = packet.decode('utf-8').strip()
            
            if not line: continue
            
            parts = line.split(',')
            if len(parts) == 10:
                sensor_values = [float(p) for p in parts]
                window.append(sensor_values)
                
                # Only run inference if our window is completely full
                if len(window) == WINDOW_SIZE:
                    raw_matrix = np.array(window)
                    clean_matrix = np.zeros_like(raw_matrix)
                    
                    # Apply DSP filters to the live window
                    for channel in range(10):
                        clean_matrix[:, channel] = apply_filters(raw_matrix[:, channel])
                    
                    # Reshape for PyTorch: [Batch, Channels, Timesteps]
                    # Transpose from 50x10 to 10x50, then add a batch dimension of 1
                    tensor_input = torch.tensor(np.transpose(clean_matrix), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    
                    # Predict!
                    with torch.no_grad(): # Don't track gradients during live testing
                        predictions = model(tensor_input)
                        probabilities = torch.softmax(predictions, dim=1)
                        confidence, predicted_class = torch.max(probabilities, 1)
                        
                        class_id = predicted_class.item()
                        conf_score = confidence.item() * 100
                        
                        # Only print if we are relatively confident and it's not idle noise
                        if class_id != 0 and conf_score > 85.0:
                            print(f"DETECTED: {CLASS_MAP[class_id]} (Confidence: {conf_score:.1f}%)")
                            
                            # Clear the window so we don't trigger 50 times for the same gesture
                            window.clear() 
                            
    except KeyboardInterrupt:
        print("\nLive testing stopped.")

if __name__ == "__main__":
    main()