import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Set the style for a professional academic look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# 1. Generate Synthetic Data (Simulating a hand movement)
# Create a smooth "Ground Truth" curve (e.g., lifting a hand)
x = np.linspace(0, 10, 100)
true_motion = np.sin(x) 

# 2. Add Noise (Simulating "Raw Model Jitter")
# Neural networks often have high-frequency noise
noise = np.random.normal(0, 0.15, 100) 
raw_output = true_motion + noise

# 3. Apply Smoothing (Simulating "SLERP/Interpolation")
# We use a Savitzky-Golay filter to mimic the effect of smoothing algorithms
smoothed_output = savgol_filter(raw_output, window_length=15, polyorder=3)

# 4. Create the Plot
plt.figure(figsize=(10, 6))

# Plot Raw Data (The "Bad" Output)
plt.plot(x, raw_output, color='#e74c3c', alpha=0.6, linewidth=1.5, label='Raw Model Output (Jittery)')

# Plot Smoothed Data (The "Good" Output)
plt.plot(x, smoothed_output, color='#2ecc71', linewidth=3, label='Smoothed Output (SLERP)')

# Add formatting
plt.title('Effect of Smoothing on Animation Quality', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Time (Frames)', fontsize=12)
plt.ylabel('Joint Rotation (Radians)', fontsize=12)
plt.legend(loc='upper right', frameon=True, framealpha=0.9, shadow=True)
plt.grid(True, linestyle='--', alpha=0.7)

# Add annotations to explain the graph
plt.annotate('High-frequency noise\n(Model Jitter)', xy=(2, 1.1), xytext=(2.5, 1.3),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('Fluid Motion\n(After Processing)', xy=(6, -0.2), xytext=(6.5, 0.2),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Save the figure
plt.tight_layout()
plt.savefig('Figure_4.8_Smoothing_Effect.png', dpi=300)
print("Chart generated successfully as 'Figure_4.8_Smoothing_Effect.png'")
plt.show()