from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar

# 加载数据
features_df = pd.read_csv("dataset/features_reduced.csv")
labels_df = pd.read_csv("dataset/labels.csv")

# 特征和标签
X = features_df
y = labels_df['state']  # 新鲜度标签作为目标变量

# 标准化数据
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# 将数据分为训练集和测试集（80%用于训练，20%用于测试）
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, random_state=42, stratify=y)

# 定义模型（随机森林）
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 使用10折StratifiedKFold进行交叉验证
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 用于存储交叉验证的得分和混淆矩阵
cv_scores = []  
all_conf_matrix = np.zeros((2, 2))  # 假设是二分类问题

# 进行交叉验证并使用进度条
for train_index, val_index in tqdm(kf.split(X_train, y_train), total=kf.get_n_splits(), desc="Cross-validation", ncols=100):
    X_kf_train, X_kf_val = X_train[train_index], X_train[val_index]
    y_kf_train, y_kf_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # 训练模型
    model.fit(X_kf_train, y_kf_train)
    y_pred = model.predict(X_kf_val)
    
    # 计算准确率并累加
    accuracy = accuracy_score(y_kf_val, y_pred)
    cv_scores.append(accuracy)
    
    # 累加混淆矩阵
    conf_matrix = confusion_matrix(y_kf_val, y_pred)
    all_conf_matrix += conf_matrix

# 计算平均交叉验证得分
cv_mean_score = np.mean(cv_scores)

# 输出交叉验证的结果
print(f"RandomForest - 平均交叉验证得分: {cv_mean_score:.4f}")
print(f"RandomForest - 累计混淆矩阵:\n{all_conf_matrix}")

# 使用完整的训练集重新训练模型
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, "model/best_random_forest_model.joblib")

# 加载模型并在测试集上评估泛化能力
loaded_model = joblib.load("model/best_random_forest_model.joblib")
y_test_pred = loaded_model.predict(X_test)

# 输出测试集的评估结果
test_accuracy = accuracy_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_report = classification_report(y_test, y_test_pred)

print("\n测试集评估结果:")
print(f"测试集准确率: {test_accuracy:.4f}")
print(f"测试集混淆矩阵:\n{conf_matrix}")
print(f"测试集分类报告:\n{class_report}")
