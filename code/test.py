import cv2
import joblib
import numpy as np
from feature_extraction import extract_features
import os

# 调整图像大小
def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)

# 转换为HSV颜色空间
def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 调整亮度和对比度
def adjust_brightness_contrast(image, alpha=1.2, beta=30):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def preprocess_image(image):
    image = resize_image(image, size=(256, 256))
    image = convert_to_hsv(image)
    image = adjust_brightness_contrast(image)
    return image

# 加载模型、scaler 和 PCA
model = joblib.load("model/best_random_forest_model.joblib")
scaler = joblib.load("model/scaler.joblib")
pca = joblib.load("model/pca_model.joblib")  # 加载已保存的 PCA

# 预测新鲜度的函数
def predict_freshness(image_path, fruit_type):
    if not os.path.exists(image_path):
        print(f"图像路径不存在: {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    image = preprocess_image(image)

    features = extract_features(image, fruit_type)
    
    # 使用 scaler 标准化
    features = scaler.transform([features])
    
    # 检查features的维度
    print(f"标准化后的特征维度: {features.shape}")

    # 确保PCA的输入是二维的
    if len(features.shape) == 1:
        features = features.reshape(1, -1)  # 将一维数据变成二维数据

    # 使用PCA进行降维
    features = pca.transform(features)
    print(f"降维后的特征维度: {features.shape}")

    prediction = model.predict(features)
    return prediction[0]

# 测试新图像
test_image_path = "dataset/test/Banana_Good.jpg"  # 替换为实际图像路径
test_fruit_type = "banana"  # 替换为图像中的水果种类

predicted_state = predict_freshness(test_image_path, test_fruit_type)
if predicted_state is not None:
    print(f"预测的新鲜度状态: {predicted_state}")
