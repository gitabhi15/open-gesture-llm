import socket
import numpy as np
import torch
import torch.nn as nn
import time
from google import genai 

# --- CONFIGURATION ---
GOOGLE_API_KEY = "AIzaSyD2939KB0sEsLKwbxtRL2c4iu3-_d7-0CM"
client = genai.Client(api_key=GOOGLE_API_KEY)

ESP32_PORT = 5005
APP_PORT = 5006  

sensor_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sensor_sock.bind(("0.0.0.0", ESP32_PORT))
sensor_sock.settimeout(1.0)

app_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- CNN ARCHITECTURE ---
class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3), nn.ReLU(),
            nn.Flatten(), nn.Linear(1472, num_classes) 
        )
    def forward(self, x): return self.net(x)

try:
    checkpoint = torch.load("model.pth", weights_only=False)
    le = checkpoint["label_encoder"]
    cnn_model = GestureCNN(len(le.classes_))
    cnn_model.load_state_dict(checkpoint["model_state"])
    cnn_model.eval()
except Exception as e:
    print(f"Error loading model.pth: {e}")
    exit()

def flush_udp():
    sensor_sock.setblocking(False)
    while True:
        try: sensor_sock.recv(1024)
        except BlockingIOError: break
    sensor_sock.setblocking(True)
    sensor_sock.settimeout(1.0)

gesture_map = {"G1": "Swipe Right", "G2": "Swipe Left", "G3": "Swipe Up", "G4": "Rotate Right"}

modes = ["PRESENTATION_MODE", "MEDIA_PLAYER_MODE"]
current_mode_idx = 0

print("🧠 LLM Brain Online! Model loaded successfully.")

while True:
    current_mode = modes[current_mode_idx]
    print(f"\n--- Current Context: {current_mode} ---")
    cmd = input("Press ENTER to gesture, 'm' to switch modes, 'q' to quit: ")
    
    if cmd.lower() == 'q': break
    if cmd.lower() == 'm':
        current_mode_idx = (current_mode_idx + 1) % len(modes)
        continue

    # --- THE FIX: 2 SECOND DELAY SO YOU CAN CLICK YOUR SLIDES ---
    print("\n⏳ You have 2 seconds to click your presentation window...")
    time.sleep(2)
    print("🔴 Listening for ESP32... DO THE GESTURE NOW!")

    data = []
    flush_udp()
    start = time.time()

    while time.time() - start < 2.0:
        try:
            packet, addr = sensor_sock.recvfrom(1024)
            line = packet.decode('utf-8').strip()
            if not line: continue
            ax, ay, az = map(float, line.split(','))
            data.append([ax, ay, az])
        except: pass

    data_arr = np.array(data)
    if len(data_arr) == 0: 
        print("No data received.")
        continue
    
    if len(data_arr) < 50:
        pad = np.repeat(data_arr[-1:], 50 - len(data_arr), axis=0)
        data_arr = np.vstack([data_arr, pad])
    else: data_arr = data_arr[:50]

    data_tensor = torch.tensor(data_arr.T).unsqueeze(0).float()
    with torch.no_grad():
        pred = torch.argmax(cnn_model(data_tensor)).item()
        gesture_name = gesture_map.get(le.inverse_transform([pred])[0], "Unknown")

    print(f"CNN Detected: {gesture_name} ... Asking LLM for intent...")

    # --- THE FIX: EXPLICITLY MAPPED "ROTATE RIGHT" FOR THE LLM ---
    prompt = f"""
    Context: The user is in {current_mode}.
    Action: The user just performed a "{gesture_name}" gesture.
    
    If in PRESENTATION_MODE: Swipe Right or Rotate Right = NEXT_SLIDE, Swipe Left = PREV_SLIDE, Swipe Up = UNKNOWN.
    If in MEDIA_PLAYER_MODE: Swipe Right or Rotate Right = NEXT_TRACK, Swipe Left = PREV_TRACK, Swipe Up = PLAY_PAUSE.
    
    Based on the context and the gesture, what is the user trying to do?
    Respond ONLY with one of the following exact output words:
    NEXT_SLIDE, PREV_SLIDE, PLAY_PAUSE, NEXT_TRACK, PREV_TRACK, VOLUME_UP, UNKNOWN.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        intent = response.text.strip()
        print(f"LLM decided intent: {intent}")
        app_sock.sendto(intent.encode('utf-8'), ("127.0.0.1", APP_PORT))
    except Exception as e:
        print(f"API Error: {e}")