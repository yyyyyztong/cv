import cv2
import pandas as pd
import numpy as np
import joblib  # 用于保存模型
from tqdm import tqdm  # 用于显示进度条
from feature_extraction import extract_features
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 加载数据集标签
data = pd.read_csv("dataset/processed_dataset_labels.csv")  # 使用预处理后的标签文件

# 定义保存特征的列表
features_list = []
labels_list = []

# 遍历数据集并显示进度条
print("提取特征中...")
for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing images"):
    image_path = row['processed_image_path']  # 使用处理后图像的路径
    state = row['state']  # 新鲜度标签
    fruit_type = row['fruit_type']  # 水果种类

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        continue  # 跳过该图像及其标签

    # 提取特征
    features = extract_features(image, fruit_type)

    # 如果特征无效（例如是NaN或全为0），则跳过该样本
    if np.any(np.isnan(features)) or np.all(features == 0):
        print(f"特征提取失败，跳过图像: {image_path}")
        continue

    # 保存有效的特征和标签
    features_list.append(features)
    labels_list.append([state, fruit_type])

# 确保特征和标签的数量一致
if len(features_list) != len(labels_list):
    print("警告：特征和标签的数量不一致！")
else:
    print("特征和标签的数量一致。")

# 转换为DataFrame
features_df = pd.DataFrame(features_list)
labels_df = pd.DataFrame(labels_list, columns=["state", "fruit_type"])

# 删除包含NaN值的行，并同步删除标签
print(f"删除NaN之前，特征数据的形状: {features_df.shape}, 标签数据的形状: {labels_df.shape}")
features_df = features_df.dropna()

# 使用features_df的索引同步删除labels_df中的行
labels_df = labels_df.loc[features_df.index]

# 检查是否仍然有NaN值
if np.any(np.isnan(features_df)):
    print("警告：特征矩阵中仍然存在NaN值！")
else:
    print("特征矩阵中没有NaN值。")

# 标准化数据
print("标准化特征数据...")
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features_df)

# 保存标准化模型
joblib.dump(scaler, "model/scaler.joblib")
print("StandardScaler 已保存为 scaler.joblib")

# 使用PCA降维
print("进行PCA降维...")
pca = PCA(n_components=0.95)  # 保留95%的方差
features_reduced = pca.fit_transform(features_standardized)

# 保存 PCA 模型
joblib.dump(pca, "model/pca_model.joblib")
print("PCA 模型已保存为 pca_model.joblib")

# 确保降维后的数据是二维的
print(f"降维后的特征形状: {features_reduced.shape}")

# 将降维后的特征保存为DataFrame，并保存到CSV文件中
print("保存降维后的特征和标签...")
features_reduced_df = pd.DataFrame(features_reduced)
features_reduced_df.to_csv("dataset/features_reduced.csv", index=False)
labels_df.to_csv("dataset/labels.csv", index=False)

print("降维后的特征和标签已保存到 features_reduced.csv 和 labels.csv")
