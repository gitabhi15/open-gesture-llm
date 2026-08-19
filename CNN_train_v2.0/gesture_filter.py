import numpy as np
import glob
import pandas as pd
import os
from scipy.signal import filtfilt, butter

DATA_DIR = "data"
OUTPUT_DIR = "training_tensors_data"
STEP_SIZE = 25
WINDOW_SIZE = 50
SAMPLING_RATE = 50.0

GESTURE_MAP = {
    "idle" : 0,
    "flick" : 1,
    "swivel" : 2,
    "wave" : 3
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_filters(column_data, cutoff_low = 0.5, cutoff_high = 50):
    nyquist = 0.5 * SAMPLING_RATE

    b_high, a_high = butter(4, cutoff_low / nyquist, btype = 'high')
    no_gravity = filtfilt(b_high, a_high, column_data)

    b_low, a_low = butter(4, cutoff_high / nyquist, bytpe = 'low')
    clean_signal = filtfilt(b_low, a_low, no_gravity)

    return clean_signal

clean_tensors = []
clean_labels = []

csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print(f"Found {len(csv_files)} recording files. Formatting...\n")

for file_path in csv_files:
    filename = os.path.basename(file_path)
    base_name = filename.split("_1")[0]

    if base_name not in GESTURE_MAP:
        print(f"Skipping {filename}: '{base_name}' not in GESTURE_MAP")
        continue

    class_id = GESTURE_MAP[base_name]
    df = pd.read(file_path)
    raw_sensors = [['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'dist']].values

    clean_sensors = np.zeros_like(raw_sensors)
    for channel in range(10):
        clean_sensors[:, channel] = apply_filters(raw_sensors[:, channel])

    num_rows = clean_sensors.shape[0]
    chunks_created = 0

    for start in range(0, num_rows - WINDOW_SIZE, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = clean_sensors[start : end, :]

        clean_tensors.append(window)
        clean_labels.append(class_id)
        chunks_created += 1

    print(f"Processed {filename} -> Generated {chunks_created} tensors (Class: {class_id})")

X = np.array(clean_tensors) 
y = np.array(clean_labels)  

# Save the final, ready-to-train binaries
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y)

print("\n=== Formatting Complete ===")
print(f"Total Model Inputs (X): {X.shape}")
print(f"Total Model Labels (y): {y.shape}")
print(f"Saved to {OUTPUT_DIR}/")