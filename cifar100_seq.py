import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Load data
data = np.load('sampleinfo/cifar100_infl_matrix.npz')
tr_mem = data['tr_mem']
tr_labels = data['tr_labels']
tt_labels = data['tt_labels']
gt_labels = np.concatenate((tr_labels, tt_labels))

# Initialize dictionaries to store indices
class_mem_0 = {i: [] for i in range(100)}  # Indices with mem value 0
class_mem_1 = {i: [] for i in range(100)}  # Indices with mem value 1

# Iterate over each class and classify indices based on mem values
for class_id in range(100):
    class_index_key = f"tr_classidx_{class_id}"
    class_indices = data[class_index_key]

    for idx in class_indices:
        if tr_mem[idx] == 0:
            class_mem_0[class_id].append(idx)
        elif tr_mem[idx] == 1:
            class_mem_1[class_id].append(idx)

# Print indices for each class
print("Each class with mem value 0 indices:")
for cls, idx_list in class_mem_0.items():
    print(f"Class {cls}: {idx_list}")

print("\nEach class with mem value 1 indices:")
for cls, idx_list in class_mem_1.items():
    print(f"Class {cls}: {idx_list}")

random_diff_class_points = {}

mem_equal_0 = list(range(0, 60000))
mem_equal_1 = list(range(0, 60000))
for i in range(60000):
    current_label = gt_labels[i]
    different_classes = [cls for cls in range(100) if cls != current_label]

    diff_class_0 = np.random.choice(different_classes)
    diff_class_1 = np.random.choice(different_classes)
    while diff_class_1 == 58 or diff_class_1 == diff_class_0:
        diff_class_1 = np.random.choice(different_classes)

    mem_0_point = np.random.choice(class_mem_0[diff_class_0])
    mem_1_point = np.random.choice(class_mem_1[diff_class_1])

    mem_equal_0[i] = mem_0_point
    mem_equal_1[i] = mem_1_point

# Save the lists mem_equal_0 and mem_equal_1
np.save('sampleinfo/mem_equal_0.npy', mem_equal_0)
np.save('sampleinfo/mem_equal_1.npy', mem_equal_1)

# Example loading of saved lists (for verification)
loaded_mem_equal_0 = np.load('sampleinfo/mem_equal_0.npy')
loaded_mem_equal_1 = np.load('sampleinfo/mem_equal_1.npy')

# Verify loaded data
print(loaded_mem_equal_0[:10])
print(loaded_mem_equal_1[:10])
