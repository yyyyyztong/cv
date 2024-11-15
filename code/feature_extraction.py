import cv2
import numpy as np

# 颜色特征提取函数
def extract_color_features(image):
    """
    计算图像的颜色直方图特征。
    将图像的HSV直方图进行归一化，并展平成一维特征向量。
    """
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # 归一化并展平成一维数组
    return hist

# 形状特征提取函数
def extract_shape_features(image, fruit_type):
    """
    根据水果类型提取适合的形状特征。
    对于香蕉，提取长宽比和方向；对于其他水果，提取面积、周长和圆度。
    """
    # 默认特征值
    default_shape_features = [0, 0, 0]  # 长宽比、方向、圆度

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contour = max(contours, key=cv2.contourArea)  # 选择最大轮廓
        
        if fruit_type == "banana":
            # 对香蕉提取长宽比和方向角度
            aspect_ratio = calculate_aspect_ratio(contour)
            orientation = calculate_orientation(contour)
            return [aspect_ratio, orientation, 0]
        
        else:
            # 对其他水果提取面积、周长和圆度
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            roundness = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter != 0 else 0
            return [area, perimeter, roundness]
    else:
        # 如果没有找到轮廓，返回默认值
        return [0, 0, 0]
    
# 辅助函数：计算长宽比
def calculate_aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / h if h != 0 else 0

# 辅助函数：计算旋转角度
def calculate_orientation(contour):
    rect = cv2.minAreaRect(contour)
    return rect[2]

# 综合特征提取函数
def extract_features(image, fruit_type):
    """
    提取图像的颜色和形状特征，组合成一个特征向量。
    """
    color_features = extract_color_features(image)
    shape_features = extract_shape_features(image, fruit_type)
    return np.concatenate([color_features, shape_features])  # 将颜色和形状特征组合成一个特征向量
