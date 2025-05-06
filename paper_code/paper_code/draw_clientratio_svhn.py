import matplotlib.pyplot as plt

# Data from the user
ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
splitmix = [71.93, 70.62, 72.49, 70.09, 70.64, 69.97, 68.48, 66.05, 54.70]
depthfl = [92.49, 92.20, 92.31, 92.36, 92.26, 91.62, 91.14, 88.20, 83.69]
heterosplit = [92.77, 92.89, 92.81, 92.58, 93.16, 92.33, 92.15, 91.58, 89.88]
fedavg_small = 68.97
fedavg_large = 92.77

# Plotting the data
plt.figure(figsize=(8, 6))
plt.plot(ratios, heterosplit, 'b-o', label="HeteroSplit")
plt.plot(ratios, depthfl, 'g-s', label="DepthFL")
plt.plot(ratios, splitmix, 'r-x', label="SplitMix")

# Adding baselines
plt.axhline(y=fedavg_small, color='k', linestyle='--', label="FedAvg Small")
plt.axhline(y=fedavg_large, color='k', linestyle='-', label="FedAvg Large")

# Adding labels and title
plt.xlabel("Ratio of Weak Devices", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.title("Accuracy of Global Model on SVHN", fontsize=20)
plt.legend(fontsize=14, framealpha=0)

# Enhancing tick labels for better clarity
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.ylim(40, 100)
# Show the chart
plt.savefig('svhn.png', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
