import matplotlib.pyplot as plt

# Data
resolutions = [100, 250, 750, 1250, 1708]
fire_relu = [254.22, 251.29, 250.62, 250.59, 249.72]
fire_siren = [266.43, 263.85, 263.19, 258.56, 258.06]
rfmid_relu = [37.29, 36.18, 36.01, 35.03, 35.04]
rfmid_siren = [74.12, 73.42, 77.55, 67.33, 67.31]

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# FIRE plot
axes[0].plot(resolutions, fire_relu, marker='o', label='ReLU')
axes[0].plot(resolutions, fire_siren, marker='s', label='SIREN')
axes[0].set_xlabel('Resolución')
axes[0].set_ylabel('Distancia Media')
axes[0].set_title('FIRE')
axes[0].grid(True)
axes[0].legend()

# RFMID plot
axes[1].plot(resolutions, rfmid_relu, marker='^', label='ReLU')
axes[1].plot(resolutions, rfmid_siren, marker='d', label='SIREN')
axes[1].set_xlabel('Resolución')
axes[1].set_ylabel('Distancia Media')
axes[1].set_title('RFMID')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig("fastchart.png")
plt.close()