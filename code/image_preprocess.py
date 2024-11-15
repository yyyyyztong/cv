import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm  # 用于显示进度条

# 读取标签文件
df = pd.read_csv("dataset/dataset_labels.csv")

# 定义存储处理后图像的文件夹路径
processed_data_path = "dataset/processed_dataset"
os.makedirs(processed_data_path, exist_ok=True)

# 定义处理后数据集的 CSV 文件路径
processed_csv_path = "dataset/processed_dataset_labels.csv"

# 调整图像大小
def resize_image(image, size=(512, 512)):
    return cv2.resize(image, size)

# 转换为HSV颜色空间
def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 调整亮度和对比度
def adjust_brightness_contrast(image, alpha=1.2, beta=30):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# 图像增强（旋转和水平翻转）
def augment_image(image):
    # 随机旋转
    angle = np.random.randint(-30, 30)
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # 随机水平翻转
    if np.random.rand() > 0.5:
        rotated = cv2.flip(rotated, 1)
        
    return rotated

# 完整的图像预处理流程
def preprocess_image(image):
    image = resize_image(image, size=(256, 256))
    image = convert_to_hsv(image)
    image = adjust_brightness_contrast(image)
    return image

# 处理和保存图像函数
def process_and_save_images():
    processed_data = []  # 用于保存处理后数据的信息

    # 使用 tqdm 为数据集添加进度条
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_path = row['image_path']
        state = row['state']
        fruit_type = row['fruit_type']
        
        # 检查图像路径是否存在
        if not os.path.exists(image_path):
            print(f"图像路径不存在: {image_path}")
            continue
        
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue

        # 图像预处理步骤
        processed_image = preprocess_image(image)

        # 创建对应的子文件夹
        processed_folder = os.path.join(processed_data_path, f"{state}_{fruit_type}")
        os.makedirs(processed_folder, exist_ok=True)

        # 构建处理后图像的保存路径
        processed_image_path = os.path.join(processed_folder, os.path.basename(image_path))
        
        # 保存预处理后的图像
        cv2.imwrite(processed_image_path, processed_image)

        # 将原始标签和处理后图像路径记录到数据集中
        processed_data.append({
            "processed_image_path": processed_image_path,
            "state": state,
            "fruit_type": fruit_type
        })

        # 对每张图像进行多个增强
        for aug_index in range(3):  # 每张图片生成3个增强版本
            augmented_image = augment_image(processed_image)
            
            # 构建增强图像的保存路径
            augmented_image_path = os.path.join(
                processed_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{aug_index}.jpg"
            )
            
            # 保存增强后的图像
            cv2.imwrite(augmented_image_path, augmented_image)

            # 将增强后的图像路径和标签记录到数据集中
            processed_data.append({
                "processed_image_path": augmented_image_path,
                "state": state,
                "fruit_type": fruit_type
            })

    # 保存处理后数据路径到 CSV 文件
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(processed_csv_path, index=False)
    print(f"处理后的图像路径已保存到 {processed_csv_path}")

# 运行图像处理和保存函数
process_and_save_images()
