import matplotlib.pyplot as plt

# Data from the user
ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#static = [74.85, 73.84, 73.43, 72.97, 71.81, 70.54, 66.53, 63.18, 56.75]
#heterosplit = [74.97, 74.62, 74.35, 74.40, 73.74, 73.73, 73.32, 72.21, 66.57]
static      = [75.61, 74.04, 73.58, 73.82, 71.87, 71.79, 67.92, 63.34, 54.26]
alternating = [75.88, 75.25, 74.32, 74.13, 73.47, 74.02, 72.30, 67.68, 62.15]
heterosplit = [75.63, 76.14, 74.82, 74.59, 73.81, 74.22, 74.27, 70.97, 65.89]
fedavg_small = 62.22
fedavg_large = 75.95

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
plt.title("Accuracy of Global Model on CIFAR10", fontsize=24)
plt.legend(fontsize=20, framealpha=0)

# Enhancing tick labels for better clarity
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.ylim(40, 80)
# Show the chart
plt.savefig('cifar10.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
