import os
import numpy as np

# 定义utility文件夹的路径
utility_dir = '.'

# 遍历utility文件夹下的每个子文件夹
for category in os.listdir(utility_dir):
    category_path = os.path.join(utility_dir, category)

    # 确保这是一个文件夹
    if os.path.isdir(category_path):
        all_data = []

        # 遍历子文件夹下的每个.npy文件
        for npy_file in os.listdir(category_path):
            if npy_file.endswith('.npy'):
                file_path = os.path.join(category_path, npy_file)

                # 读取npy文件
                data = np.load(file_path)
                all_data.append(data)

        # 计算该文件夹下所有.npy文件的平均值
        if all_data:
            average_data = np.mean(all_data, axis=0)

            # 保存平均值到当前目录
            save_path = f'{category}_average.npy'
            np.save(save_path, average_data)
            print(f"Saved average for {category} to {save_path}")
