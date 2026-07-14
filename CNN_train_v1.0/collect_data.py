import socket
import numpy as np
import time
import os
import sys

UDP_IP = "0.0.0.0"  # Listen on all available network interfaces
UDP_PORT = 5005     # Must match the port in the Arduino code

# Set up the UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0) # Don't block forever if ESP32 turns off

print(f"Listening for UDP data on port {UDP_PORT}...")

os.makedirs("data", exist_ok=True)
gesture = input("Enter gesture label (e.g., G1): ")
SAMPLE_DURATION = 2  # seconds

def flush_udp_buffer():
    """Clears out old network packets so we only record fresh movement."""
    sock.setblocking(False)
    while True:
        try:
            sock.recv(1024)
        except BlockingIOError:
            break
    sock.setblocking(True)
    sock.settimeout(1.0)

while True:
    cmd = input("\nPress ENTER to record sample (or type 'q' to quit): ")

    if cmd.lower() == 'q':
        print("Exiting...")
        break

    print("Get ready...")
    time.sleep(1)

    print("Recording NOW!")
    data = []
    
    flush_udp_buffer()
    start = time.time()

    while time.time() - start < SAMPLE_DURATION:
        try:
            # Receive UDP packet
            packet, addr = sock.recvfrom(1024)
            line = packet.decode('utf-8').strip()
            
            if not line: continue
            ax, ay, az = map(float, line.split(','))
            data.append([ax, ay, az])
        except socket.timeout:
            pass # Ignore timeouts
        except Exception as e:
            pass # Ignore malformed packets

    data = np.array(data)

    if len(data) > 0:
        filename = f"data/{gesture}_{int(time.time())}.npy"
        np.save(filename, data)
        print(f"Saved sample: {filename}")
        print(f"Samples collected: {len(data)} timesteps")
    else:
        print("ERROR: No data received! Is the ESP32 on the same Wi-Fi?")