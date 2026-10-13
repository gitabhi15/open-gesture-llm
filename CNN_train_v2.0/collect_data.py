import socket
import csv
import time
import os

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.1) 

os.makedirs("data", exist_ok=True)

print("--- V3.0 Batch Data Logger ---")
session_name = input("Enter the gesture name (e.g., flick, swivel, wave, idle): ").strip()
safe_session_name = "".join(c for c in session_name if c.isalnum() or c in ('_', '-')) or "unnamed"

HEADER = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz", "dist"]
RECORD_DURATION = 4.5  

print(f"\nGesture set to: {safe_session_name}")
print(f"Recording window: {RECORD_DURATION} seconds per file.")
print("Press Ctrl+C at any time to exit the program.\n")

count = 1
try:
    while True:
        input(f"[{count}] Press ENTER to record...")
        
        # 1. Drain the stale UDP buffer to avoid capturing old movements
        while True:
            try:
                sock.recvfrom(1024)
            except socket.timeout:
                break
        
        # 2. Restore normal timeout for the active recording phase
        sock.settimeout(1.0)
        
        session_filename = f"data/{safe_session_name}_{int(time.time())}.csv"
        print("RECORDING (Rest -> Snap -> Rest)...", end="", flush=True)
        
        start_time = time.time()
        samples_collected = 0
        
        with open(session_filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(HEADER)
            
            while (time.time() - start_time) < RECORD_DURATION:
                try:
                    packet, _ = sock.recvfrom(1024)
                    line = packet.decode('utf-8').strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) == 10:
                            writer.writerow([time.time()] + [float(p) for p in parts])
                            samples_collected += 1
                except (socket.timeout, ValueError):
                    pass
        
        print(f"  Saved {samples_collected} samples.")
        count += 1
        
        # Reset timeout for the next buffer drain
        sock.settimeout(0.1) 

except KeyboardInterrupt:
    print(f"\n\nBatch recording session ended. Total files saved: {count - 1}")