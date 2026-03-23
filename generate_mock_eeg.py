import csv
import math
import random

# Settings
SAMPLE_RATE = 128 # Hz
DURATION = 30 # seconds
TOTAL_SAMPLES = SAMPLE_RATE * DURATION

CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1"]

# Generate data
rows = []
for i in range(TOTAL_SAMPLES):
    time_sec = i / SAMPLE_RATE
    
    # We'll simulate a base wave plus some noise for each channel
    # Simulate an alpha wave (10 Hz) + beta wave (20 Hz) + delta wave (2 Hz)
    
    row = {"time": f"{time_sec:.3f}"}
    for ch in CHANNELS:
        # Base low freq oscillation
        delta = 20 * math.sin(2 * math.pi * 2 * time_sec + random.random() * 0.1)
        
        # Alpha wave (10 Hz)
        alpha = 15 * math.sin(2 * math.pi * 10 * time_sec + random.random() * 0.1)
        
        # Beta wave (20 Hz)
        beta = 5 * math.sin(2 * math.pi * 20 * time_sec + random.random() * 0.1)
        
        # Add some random noise and a baseline offset
        val = 40.0 + delta + alpha + beta + (random.random() * 5.0 - 2.5)
        
        row[ch] = f"{val:.3f}"
        
    rows.append(row)

# Write to CSV
with open("sample_eeg_recording.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["time"] + CHANNELS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {TOTAL_SAMPLES} rows ({DURATION} seconds) of EEG data.")
