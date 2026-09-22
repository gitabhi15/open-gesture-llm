import numpy as np
import glob
import pandas as pd
import os
from scipy.signal import filtfilt, butter

DATA_DIR = "data"
OUTPUT_DIR = "training_tensors_data"
SAMPLING_RATE = 50.0

GESTURE_MAP = {"idle": 0, "flick": 1, "swivel": 2, "wave": 3}
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_filters(column_data):
    nyquist = 0.5 * SAMPLING_RATE
    b_high, a_high = butter(4, 0.5 / nyquist, btype='high')
    no_gravity = filtfilt(b_high, a_high, column_data)
    
    b_low, a_low = butter(4, 5.0 / nyquist, btype='low')
    return filtfilt(b_low, a_low, no_gravity)

clean_tensors = []
clean_labels = []

csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

for file_path in csv_files:
    filename = os.path.basename(file_path)
    base_name = filename.split("_1")[0]

    if base_name not in GESTURE_MAP:
        continue

    df = pd.read_csv(file_path)
    raw_sensors = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'dist']].values

    clean_sensors = np.zeros_like(raw_sensors)
    for channel in range(10):
        clean_sensors[:, channel] = apply_filters(raw_sensors[:, channel])

    # Trigger-centered windowing: Find the highest gyroscope energy spike
    gyro_mags = np.sqrt(np.sum(raw_sensors[:, 3:6]**2, axis=1))
    peak_idx = np.argmax(gyro_mags)
    
    start = peak_idx - 10
    end = peak_idx + 40
    
    if start >= 0 and end <= len(clean_sensors):
        clean_tensors.append(clean_sensors[start:end, :])
        clean_labels.append(GESTURE_MAP[base_name])

X = np.array(clean_tensors) 
y = np.array(clean_labels)  

# Z-score Normalization
mean_vals = np.mean(X, axis=(0, 1), keepdims=True)
std_vals = np.std(X, axis=(0, 1), keepdims=True)
std_vals[std_vals == 0] = 1.0 
X = (X - mean_vals) / std_vals

np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y)
np.save(os.path.join(OUTPUT_DIR, "mean.npy"), mean_vals)
np.save(os.path.join(OUTPUT_DIR, "std.npy"), std_vals)