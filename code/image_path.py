import os
import pandas as pd

# 定义数据集文件夹路径
dataset_path = "dataset/original_dataset"

# 打印当前目录
print("Current directory:", os.getcwd())

# 创建列表存储文件路径和标签
data = []

# 遍历每个文件夹
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    
    # 确认是文件夹
    if os.path.isdir(folder_path):
        # 分割文件夹名称，提取状态和水果种类
        parts = folder_name.split('_')
        if len(parts) == 2:
            state = parts[0]  # 状态标签（如fresh或stale）
            fruit_type = parts[1]  # 水果种类（如apple、banana、orange）
            
            # 遍历文件夹中的每张图像
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                # 将路径、状态和水果种类标签添加到列表
                data.append([file_path, state, fruit_type])

# 转换为DataFrame
df = pd.DataFrame(data, columns=["image_path", "state", "fruit_type"])

# 保存为CSV文件
df.to_csv("dataset/dataset_labels.csv", index=False)
print("标签数据已保存到dataset_labels.csv文件中")
