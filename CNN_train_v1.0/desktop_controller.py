import socket
import subprocess

# Listen on localhost
LOCAL_UDP_IP = "127.0.0.1"
LOCAL_UDP_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LOCAL_UDP_IP, LOCAL_UDP_PORT))

print("=" * 40)
print(" 💻 DESKTOP CONTROLLER APP RUNNING (Linux Native)")
print("=" * 40)
print("Listening for LLM intents on port 5006...\n")

def press_key(key):
    """Safely runs the Linux xdotool command to simulate a keypress"""
    try:
        subprocess.run(["xdotool", "key", key], check=True)
    except FileNotFoundError:
        print("ERROR: xdotool is not installed. Run: sudo dnf install xdotool")
    except Exception as e:
        print(f"Failed to press {key}: {e}")

while True:
    try:
        packet, addr = sock.recvfrom(1024)
        intent = packet.decode('utf-8').strip()
        print(f"[RECEIVED INTENT]: {intent}")

        # Map the LLM's intent to Linux xdotool commands
        if intent == "NEXT_SLIDE":
            print("   -> Pressing RIGHT ARROW")
            press_key("Right")
            
        elif intent == "PREV_SLIDE":
            print("   -> Pressing LEFT ARROW")
            press_key("Left")
            
        elif intent == "PLAY_PAUSE":
            print("   -> Pressing SPACEBAR")
            press_key("space")
            
        elif intent == "VOLUME_UP":
            print("   -> Pressing VOLUME UP")
            # This is the native Linux multimedia key for Volume Up
            press_key("XF86AudioRaiseVolume") 
            
        elif intent == "UNKNOWN":
            print("   -> Ignored (LLM determined gesture doesn't match context)")
            
    except Exception as e:
        print(f"Error: {e}")