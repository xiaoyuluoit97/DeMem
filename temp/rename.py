import os
import numpy as np
import pandas as pd

# Define the directory containing the npy files
directory = "/cifar100"

# List all npy files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.npy')]

# Initialize a dictionary to store the variances
variances = {}

# Read each npy file, calculate the variance, and store the data
for file in files:
    parts = file.split('_')
    if parts[1] == 'fat':
        category = parts[0] + '_' + parts[1]  # Merge the 'fat' part with the first part
    else:
        category = parts[0]  # Extract the category from the filename
    if category not in variances:
        variances[category] = []
    file_path = os.path.join(directory, file)
    accuracy_list = np.load(file_path).tolist()  # Read the npy file and convert to list
    variance = np.var(accuracy_list)  # Calculate the variance
    variances[category].append(variance)

# Calculate the overall mean variance for each category
mean_variances = {category: np.mean(var_list) for category, var_list in variances.items()}

# Print the overall mean variances
for category, mean_variance in mean_variances.items():
    print(f"Category: {category}, Overall Mean Variance: {mean_variance}")
