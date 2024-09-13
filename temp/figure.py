import os
import numpy as np
AUGTYPE = "pgdat_reg_8_0.6"
# Define the directory containing the npy files
directory = "acc/cifar100/clean"
directory = "/Users/xiaoyuluo/workspace/privacy_and_aug-main/acc/mia/lira"
directory = f"/home/luo/acc/mia/lira/TPR/%s/"%(AUGTYPE)

# List all npy files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.npy')]

# Initialize dictionaries to store the variances and mean accuracies
variances = {}
mean_accuracies = {}
each_class_utility = {}
# Read each npy file, calculate the variance and mean, and store the data
THROW = "class.npy"
for file in files:
    parts = file.split('_')
    print(parts[-1])
    if parts[-1] == THROW:
        print("no want class")
        continue

    if parts[1] == 'fat' or parts[1] == 'reg':
        category = parts[0] + '_' + parts[1]  # Merge the 'fat' part with the first part
    elif parts[-2]== "tpr0001fpr":
        category = parts[-2]
    elif parts[-2]== "tpr01fpr":
        category = parts[-2]
    else:
        category = parts[0]  # Extract the category from the filename

    if category not in variances:
        variances[category] = []
    if category not in mean_accuracies:
        mean_accuracies[category] = []
    if category not in each_class_utility:
        each_class_utility[category] = []

    file_path = os.path.join(directory, file)
    accuracy_list = np.load(file_path).tolist()  # Read the npy file and convert to list
    variance = np.var(accuracy_list)  # Calculate the variance
    mean_accuracy = np.mean(accuracy_list)  # Calculate the mean accuracy

    variances[category].append(variance)
    mean_accuracies[category].append(mean_accuracy)
    each_class_utility[category].append(accuracy_list)



# Calculate the overall mean variance and mean accuracy for each category
mean_variances = {category: np.mean(var_list) for category, var_list in variances.items()}
overall_mean_accuracies = {category: np.mean(acc_list) for category, acc_list in mean_accuracies.items()}


averaged_class_utility = {}
for key, list_of_lists in each_class_utility.items():
    # 转换为 NumPy 数组以便于操作
    array_of_lists = np.array(list_of_lists)
    # 计算元素逐项平均
    averaged_list = np.mean(array_of_lists, axis=0)
    # 存储平均后的结果
    averaged_class_utility[key] = averaged_list.tolist()

    np.save("result/%s_tpracc_%s_memwise"%(AUGTYPE,key),averaged_list.tolist())

# Print the overall mean variances
print("Overall Mean Variances:")
for category, mean_variance in mean_variances.items():
    print(f"Category: {category}, Overall Mean Variance: {mean_variance}")

# Print the overall mean accuracies
print("\nOverall Mean Accuracies:")
for category, overall_mean_accuracy in overall_mean_accuracies.items():
    print(f"Category: {category}, Overall Mean Accuracy: {overall_mean_accuracy}")
