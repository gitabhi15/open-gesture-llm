import socket

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"📡 Wireless Serial Monitor listening on port {UDP_PORT}...\n")

while True:
    try:
        packet, addr = sock.recvfrom(1024)
        line = packet.decode('utf-8').strip()
        print(line)
    except KeyboardInterrupt:
        print("\nExiting monitor...")
        break