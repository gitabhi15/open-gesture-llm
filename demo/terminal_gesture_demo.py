from . import demo_gesture
from .demo_gesture import *

print("Starting gesture demo (Terminal)...")

while True:
    try:
        line = ser.readline().decode().strip()
        ax, ay, az = map(float, line.split(','))

        update_buffer([ax, ay, az])

        if len(buffer) == 20:
            features = extract_features(buffer)
            token = tokenize(features)
            update_tokens(token)
            intent = interpret(tokens)

            print("\n---")
            print(f"RAW: {ax:.2f}, {ay:.2f}, {az:.2f}")
            print(f"TOKEN: {token}")
            print(f"SEQ: {' '.join(tokens)}")
            print(f"INTENT: {intent}")

    except Exception as e:
        print("Error:", e)