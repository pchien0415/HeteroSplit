import matplotlib.pyplot as plt

# Data from the user
ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
splitmix = [26.40, 26.87, 27.05, 26.36, 25.70, 24.10, 22.23, 19.79, 15.67]
depthfl = [48.38, 47.21, 46.04, 44.66, 43.57, 40.97, 37.85, 33.37, 25.56]
proposed = [49.85, 49.27, 49.36, 47.92, 48.74, 48.03, 47.32, 45.37, 40.44]
fedavg_small = 25.25
fedavg_large = 49.77

# Plotting the data
plt.figure(figsize=(8, 6))
plt.plot(ratios, proposed, 'b-o', label="HeteroSplit")
plt.plot(ratios, depthfl, 'g-s', label="DepthFL")
plt.plot(ratios, splitmix, 'r-x', label="SplitMix")

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
