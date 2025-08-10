# VISUALIZATION of the RESPONSE - FIXED VERSION
# This script fixes the dimension mismatch issue

import numpy as np
import matplotlib.pyplot as plt

# Load the response data
RESPONSE_AV = np.load('../DATA/GEN_RES/RESPONSE_AV.npy')
RESPONSE_EQ = np.load('../DATA/GEN_RES/RESPONSE_EQ.npy')
RESPONSE_IP = np.load('../DATA/GEN_RES/RESPONSE_IP.npy')
RESPONSE_SW = np.load('../DATA/GEN_RES/RESPONSE_SW.npy')

# Print data shapes to verify
print("Data shapes:")
print(f"RESPONSE_AV shape: {RESPONSE_AV.shape}")
print(f"RESPONSE_EQ shape: {RESPONSE_EQ.shape}")
print(f"RESPONSE_IP shape: {RESPONSE_IP.shape}")
print(f"RESPONSE_SW shape: {RESPONSE_SW.shape}")

# determine an index to check
index_1 = 1
index_2 = 1

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 计算时间轴 - FIXED: 使用正确的长度6000个点
signal_length = RESPONSE_AV.shape[2]  # 6000
dt = 0.01
time_axis = np.arange(0, signal_length * dt, dt)  # 0到60秒，步长0.01秒，共6000个点

print(f"Time axis length: {len(time_axis)}")
print(f"Expected length: {signal_length}")

# subplot to show the response signals, each subplot contains all the DOFs
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Response Signals Example', fontsize=16, y=0.98)

# AMBIENT VIBRATION
axes[0, 0].plot(time_axis, RESPONSE_AV[index_1, :, :].T, alpha=0.7, linewidth=1.2)
axes[0, 0].set_title('Ambient Vibration', pad=10, fontsize=14)
axes[0, 0].set_xlabel('Time (s)', fontsize=16)
axes[0, 0].set_ylabel('Amplitude (g)', fontsize=16)
axes[0, 0].tick_params(axis='both', which='major', labelsize=14)
axes[0, 0].grid(True, alpha=0.3)

# EARTHQUAKE (右上角 - 不添加图例)
axes[0, 1].plot(time_axis, RESPONSE_EQ[index_1, :, :].T, alpha=0.7, linewidth=1.2)
axes[0, 1].set_title('Earthquake', pad=10, fontsize=14)
axes[0, 1].set_xlabel('Time (s)', fontsize=16)
axes[0, 1].set_ylabel('Amplitude (g)', fontsize=16)
axes[0, 1].tick_params(axis='both', which='major', labelsize=14)
axes[0, 1].grid(True, alpha=0.3)

# IMPACT (左下角 - 添加图例)
lines_ip = axes[1, 0].plot(time_axis, RESPONSE_IP[index_1, :, :].T, alpha=0.7, linewidth=1.2)
axes[1, 0].set_title('Impact', pad=10, fontsize=14)
axes[1, 0].set_xlabel('Time (s)', fontsize=16)
axes[1, 0].set_ylabel('Amplitude (g)', fontsize=16)
axes[1, 0].tick_params(axis='both', which='major', labelsize=14)
axes[1, 0].grid(True, alpha=0.3)
# 只在左下角添加图例，显示DOF编号
legend_labels = [f'DOF {i+1}' for i in range(RESPONSE_IP.shape[2])]
axes[1, 0].legend(legend_labels, loc='upper right', fontsize=12)

# STRONG WIND
axes[1, 1].plot(time_axis, RESPONSE_SW[index_1, :, :].T, alpha=0.7, linewidth=1.2)
axes[1, 1].set_title('Strong Wind', pad=10, fontsize=14)
axes[1, 1].set_xlabel('Time (s)', fontsize=16)
axes[1, 1].set_ylabel('Amplitude (g)', fontsize=16)
axes[1, 1].tick_params(axis='both', which='major', labelsize=14)
axes[1, 1].grid(True, alpha=0.3)

# Adjust layout to be more compact
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)

# Save high-resolution image BEFORE showing
plt.savefig('response_signals_example_fixed.jpg', 
            dpi=600, 
            bbox_inches='tight', 
            facecolor='white', 
            edgecolor='none',
            format='jpg')
print("High-resolution image saved as 'response_signals_example_fixed.jpg'")

# Display the plot AFTER saving
plt.show()
