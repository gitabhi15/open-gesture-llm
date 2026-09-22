import os
import json
import time
import socket
import threading
import subprocess
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from scipy.signal import butter, lfilter
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

UDP_IP = "0.0.0.0"
UDP_PORT = 5005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_MAP = {0: "idle", 1: "flick", 2: "swivel", 3: "wave"}
CACHE_FILE = "llm_cache.json"
MOTION_THRESHOLD = 300.0 

API_KEY = os.environ.get("GROQ_API_KEY", "")

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f: return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f: json.dump(cache, f, indent=4)

llm_cache = load_cache()

mean_vals = np.load("training_tensors_data/mean.npy").squeeze()
std_vals = np.load("training_tensors_data/std.npy").squeeze()

raw_window = deque(maxlen=50)
filtered_window = deque(maxlen=50)

nyq = 25.0
b_hi, a_hi = butter(4, 0.5/nyq, btype='high')
b_lo, a_lo = butter(4, 5.0/nyq, btype='low')
zi_hi = np.zeros((10, max(len(a_hi), len(b_hi)) - 1))
zi_lo = np.zeros((10, max(len(a_lo), len(b_lo)) - 1))

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

def udp_listener():
    global zi_hi, zi_lo
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    while True:
        packet, _ = sock.recvfrom(1024)
        parts = packet.decode('utf-8').strip().split(',')
        if len(parts) == 10:
            raw_sample = np.array([float(p) for p in parts])
            raw_window.append(raw_sample)
            
            clean_sample = np.zeros(10)
            for i in range(10):
                val, zi_hi[i] = lfilter(b_hi, a_hi, [raw_sample[i]], zi=zi_hi[i])
                val, zi_lo[i] = lfilter(b_lo, a_lo, val, zi=zi_lo[i])
                clean_sample[i] = val[0]
            filtered_window.append(clean_sample)

def execute_semantic_action(gesture_name):
    try:
        dbus_cmd = [
            "gdbus", "call", "--session", 
            "--dest", "org.gnome.Shell", 
            "--object-path", "/org/gnome/Shell/Extensions/WindowsExt", 
            "--method", "org.gnome.Shell.Extensions.WindowsExt.FocusTitle"
        ]
        active_window = subprocess.check_output(dbus_cmd).decode('utf-8').split("'")[1].lower()
    except Exception:
        active_window = "unknown"

    cache_key = f"{gesture_name}_{active_window}"
    
    if cache_key in llm_cache:
        subprocess.run(llm_cache[cache_key], shell=True)
        return

    system_prompt = """
    You are a Linux Wayland system controller translating physical gestures into terminal commands.
    Tools:
    - Media: playerctl play-pause, playerctl position 10+, playerctl position 10-, playerctl next, playerctl previous
    
    Rules: Return strictly raw JSON with keys "reasoning" and "command". Do not use markdown.
    """
    
    try:
        client = Groq(api_key=API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Gesture: {gesture_name}\nActive Window: {active_window}"}
            ],
            model="openai/gpt-oss-20b",
        )
        
        result_text = chat_completion.choices[0].message.content
        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            
        result = json.loads(result_text)
        command = result.get('command')
        
        if command and command.lower() != "none":
            llm_cache[cache_key] = command
            save_cache(llm_cache)
            subprocess.run(command, shell=True)
            
    except Exception as e:
        print(f"API Error: {e}")

def main():
    model = GestureNet(4).to(DEVICE)
    model.load_state_dict(torch.load("gesture_model.pth", map_location=DEVICE))
    model.eval()

    threading.Thread(target=udp_listener, daemon=True).start()
    
    last_trigger_time = 0
    capture_countdown = 0

    while True:
        if len(raw_window) == 50:
            current_raw = raw_window[-1]
            gyro_mag = np.sqrt(current_raw[3]**2 + current_raw[4]**2 + current_raw[5]**2)
            
            if gyro_mag > MOTION_THRESHOLD and capture_countdown == 0 and (time.time() - last_trigger_time > 0.8):
                capture_countdown = 40 
                
            if capture_countdown > 0:
                capture_countdown -= 1
                if capture_countdown == 0:
                    clean_matrix = np.array(filtered_window)
                    norm_matrix = (clean_matrix - mean_vals) / std_vals
                    
                    tensor_input = torch.tensor(norm_matrix.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        probs = torch.softmax(model(tensor_input), dim=1)
                        conf, pred = torch.max(probs, 1)
                        class_id = pred.item()
                        conf_score = conf.item() * 100
                        
                        if class_id != 0 and conf_score > 85.0:
                            print(f"\nDETECTED: {CLASS_MAP[class_id]} ({conf_score:.1f}%)")
                            execute_semantic_action(CLASS_MAP[class_id])
                            
                    last_trigger_time = time.time()
        time.sleep(0.005)

if __name__ == "__main__":
    main()