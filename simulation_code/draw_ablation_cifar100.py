import matplotlib.pyplot as plt

# Data from the user
ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
static = [48.38, 47.21, 46.04, 44.66, 43.57, 40.97, 37.85, 33.37, 25.56]
alternating = [49.07, 47.90, 48.38, 47.02, 46.39, 45.90, 44.18, 42.38, 36.41]
heterosplit = [49.85, 49.27, 49.36, 47.92, 48.74, 48.03, 47.32, 45.37, 40.44]
fedavg_small = 25.25
fedavg_large = 49.77

# Plotting the data
plt.figure(figsize=(8, 6))
plt.plot(ratios, heterosplit, 'b-o', label="HeteroSplit")
plt.plot(ratios, alternating, 'r-x', label="Alternating")
plt.plot(ratios, static, 'g-s', label="Static")

# Adding baselines
plt.axhline(y=fedavg_small, color='k', linestyle='--', label="FedAvg Small")
plt.axhline(y=fedavg_large, color='k', linestyle='-', label="FedAvg Large")

# Adding labels and title
plt.xlabel("Ratio of Weak Devices", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("Accuracy of Global Model on CIFAR100", fontsize=24)
plt.legend(fontsize=18, framealpha=0)

# Enhancing tick labels for better clarity
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.ylim(0, 55)
# Show the chart
plt.savefig('cifar100.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
