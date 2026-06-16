import serial
import numpy as np

# CHANGE THIS
PORT = '/dev/ttyUSB0'   # or '/dev/ttyUSB0'
BAUD = 115200

ser = serial.Serial(PORT, BAUD)

buffer = []
tokens = []

def update_buffer(x):
    buffer.append(x)
    if len(buffer) > 20:
        buffer.pop(0)

def extract_features(buffer):
    arr = np.array(buffer)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return np.concatenate([mean, std])

def tokenize(features):
    ax, ay, az, *_ = features
    
    if ax > 3:
        return "G1"   # right
    elif ax < -3:
        return "G2"   # left
    elif ay > 3:
        return "G3"   # up
    else:
        return "G0"

def update_tokens(t):
    tokens.append(t)
    if len(tokens) > 5:
        tokens.pop(0)

def interpret(tokens):
    seq = " ".join(tokens)
    
    if "G1 G1" in seq:
        return "NEXT"
    elif "G2 G2" in seq:
        return "PREVIOUS"
    elif "G3" in seq:
        return "ZOOM"
    else:
        return "IDLE"