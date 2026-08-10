import socket
import csv
import time
import os

UDP_IP = "0.0.0.0"  # Listen on all available network interfaces
UDP_PORT = 5005     # Must match the port in the Arduino code

# Set up the UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0) # Don't block forever if ESP32 turns off

os.makedirs("data", exist_ok=True)

print("--- V2.0 Continuous Data Logger ---")
# Prompt the user for the specific gesture/session being recorded
session_name = input("Enter the gesture or session name (e.g., idle_noise, swipe_left, flick): ").strip()

# Sanitize input to ensure a valid filename
safe_session_name = "".join(c for c in session_name if c.isalnum() or c in ('_', '-'))
if not safe_session_name:
    safe_session_name = "unnamed_session"

# Generate the labeled, timestamped continuous log file
session_filename = f"data/{safe_session_name}_{int(time.time())}.csv"

# Header matching the unified Arduino CSV output (No label column needed for Method 1)
HEADER = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz", "dist"]

print(f"\nListening for UDP data on port {UDP_PORT}...")
print(f"Logging multi-sensor continuous stream to: {session_filename}")
print("Press Ctrl+C to stop recording.\n")

sample_count = 0

try:
    with open(session_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(HEADER)  # Write header row
        
        while True:
            try:
                # Receive UDP packet
                packet, _ = sock.recvfrom(1024)
                line = packet.decode('utf-8').strip()
                
                if not line:
                    continue
                
                parts = line.split(',')
                
                # Verify packet contains all 10 expected values
                if len(parts) == 10:
                    sensor_values = [float(p) for p in parts]
                    
                    # Prepend local system timestamp
                    row = [time.time()] + sensor_values
                    writer.writerow(row)
                    
                    sample_count += 1
                    
                    # Periodically flush buffer and update terminal
                    if sample_count % 100 == 0:
                        csv_file.flush()
                        print(f"Logged {sample_count} samples...", end='\r')
                        
            except socket.timeout:
                pass  # Keep waiting for packets
            except ValueError:
                pass  # Ignore malformed packets during stream initialization

except KeyboardInterrupt:
    print(f"\n\nRecording stopped successfully.")
    print(f"Total samples saved: {sample_count}")
    print(f"Output saved to: {session_filename}")