import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- PARAMETERS ---
filename = "AFE.txt"
ui = 20e-9  # unit interval = 20 ns
ycol = "V(buffer_out)"  # CHANGED to analog signal
offset = 0.0

# --- LOAD DATA ---
df = pd.read_csv(filename, sep="\t")
t = df["time"].to_numpy(dtype=float)
y = df[ycol].to_numpy(dtype=float)

# --- FOLD INTO EYE ---
tm = (t - offset) % ui

# --- EXTRACT SAMPLING POINT (middle of UI) ---
# Find samples near the center of the bit period (0.5 UI)
sample_window = 0.1 * ui  # ±10% window around center
mask = (tm > 0.5*ui - sample_window) & (tm < 0.5*ui + sample_window)
y_samples = y[mask]

# --- SEPARATE LOGIC LEVELS ---
# Use k-means or simple threshold
threshold = np.median(y_samples)
y0 = y_samples[y_samples < threshold]  # logic 0 samples
y1 = y_samples[y_samples >= threshold]  # logic 1 samples

# --- CALCULATE STATISTICS ---
mu0 = np.mean(y0)
mu1 = np.mean(y1)
sigma0 = np.std(y0)
sigma1 = np.std(y1)
threshold_optimal = (mu0 * sigma1 + mu1 * sigma0) / (sigma0 + sigma1)

# --- Q FACTOR ---
Q = (mu1 - mu0) / (sigma0 + sigma1)

# --- BER ESTIMATE (assuming Gaussian noise) ---
from scipy.special import erfc
BER = 0.5 * erfc(Q / np.sqrt(2))

# --- PRINT RESULTS ---
print(f"Eye Diagram Analysis:")
print(f"  Logic 0: μ₀ = {mu0*1e3:.3f} mV, σ₀ = {sigma0*1e6:.1f} µV")
print(f"  Logic 1: μ₁ = {mu1*1e3:.3f} mV, σ₁ = {sigma1*1e6:.1f} µV")
print(f"  Eye height: {(mu1-mu0)*1e3:.3f} mV")
print(f"  Q factor: {Q:.2f}")
print(f"  Estimated BER: {BER:.2e}")

# --- TIMING JITTER ANALYSIS ---
# Find zero crossings (transitions through threshold)
crossings = []
for i in range(1, len(y)):
    if (y[i-1] < threshold and y[i] >= threshold) or (y[i-1] >= threshold and y[i] < threshold):
        # Linear interpolation to find exact crossing time
        t_cross = t[i-1] + (threshold - y[i-1]) * (t[i] - t[i-1]) / (y[i] - y[i-1])
        crossings.append(t_cross % ui)

if len(crossings) > 10:
    # Crossings should ideally be at 0 or ui/2
    # Separate rising and falling edges
    crossings = np.array(crossings)
    # Fold around expected crossing points
    jitter_early = crossings[crossings < ui/4]  # near 0
    jitter_late = crossings[crossings > 3*ui/4] - ui  # near ui (wrap to 0)
    jitter_mid = crossings[(crossings > ui/4) & (crossings < 3*ui/4)] - ui/2  # near ui/2
    
    all_jitter = np.concatenate([jitter_early, jitter_late, jitter_mid])
    rms_jitter = np.std(all_jitter)
    pp_jitter = np.max(all_jitter) - np.min(all_jitter)
    
    print(f"\nTiming Analysis:")
    print(f"  RMS jitter: {rms_jitter*1e12:.1f} ps")
    print(f"  Peak-peak jitter: {pp_jitter*1e12:.1f} ps")

# --- ENHANCED EYE PLOT ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Eye diagram
ax1.plot(tm*1e9, y, ".", markersize=0.5, alpha=0.3)
ax1.axhline(mu0, color='r', linestyle='--', label=f'μ₀={mu0*1e3:.2f}mV')
ax1.axhline(mu1, color='g', linestyle='--', label=f'μ₁={mu1*1e3:.2f}mV')
ax1.axhline(threshold_optimal, color='k', linestyle=':', label='Threshold')
ax1.fill_between([0, ui*1e9], mu0-sigma0, mu0+sigma0, alpha=0.2, color='r')
ax1.fill_between([0, ui*1e9], mu1-sigma1, mu1+sigma1, alpha=0.2, color='g')
ax1.set_xlabel("Time within UI (ns)")
ax1.set_ylabel("Amplitude (V)")
ax1.set_title(f"Eye Diagram (Q={Q:.2f}, BER≈{BER:.1e})")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Histogram at sampling point
ax2.hist(y0*1e3, bins=50, alpha=0.5, label='Logic 0', color='r')
ax2.hist(y1*1e3, bins=50, alpha=0.5, label='Logic 1', color='g')
ax2.axvline(threshold*1e3, color='k', linestyle=':', label='Threshold')
ax2.set_xlabel("Voltage (mV)")
ax2.set_ylabel("Count")
ax2.set_title("Voltage Distribution at Sampling Point")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("eye_analysis.png", dpi=150)
