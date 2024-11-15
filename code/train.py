from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# 加载数据
features_df = pd.read_csv("dataset/features_reduced.csv")
labels_df = pd.read_csv("dataset/labels.csv")

# 特征和标签
X = features_df
y = labels_df['state']  # 新鲜度标签作为目标变量

# 标准化数据
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# 定义模型
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

# 使用10折StratifiedKFold进行交叉验证
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 用于存储每个模型的结果
results = {}

for model_name, model in models.items():
    cv_scores = []  # 用于存储每次交叉验证的得分
    all_conf_matrix = np.zeros((2, 2))  # 用于存储所有折的混淆矩阵（假设是二分类问题）
    all_class_report = ""  # 用于存储分类报告
    
    # 进行交叉验证
    for train_index, val_index in kf.split(X_standardized, y):
        X_train, X_val = X_standardized[train_index], X_standardized[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # 训练模型
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # 计算准确率
        accuracy = accuracy_score(y_val, y_pred)
        cv_scores.append(accuracy)
        
        # 累加混淆矩阵
        conf_matrix = confusion_matrix(y_val, y_pred)
        all_conf_matrix += conf_matrix
        
        # 分类报告
        class_report = classification_report(y_val, y_pred)
        all_class_report += class_report
    
    # 存储模型的平均交叉验证得分
    results[model_name] = {
        "cv_mean_score": np.mean(cv_scores),
        "confusion_matrix": all_conf_matrix,
        "classification_report": all_class_report
    }
    
    # 输出平均交叉验证得分
    print(f"{model_name} - 平均交叉验证得分: {results[model_name]['cv_mean_score']:.4f}")
    
# 打印每个模型的最终评估结果（混淆矩阵和分类报告）
print("\n模型评估结果:")
for model_name, result in results.items():
    print(f"\n{model_name} - 混淆矩阵:\n{result['confusion_matrix']}")
    print(f"{model_name} - 分类报告:\n{result['classification_report']}")

# 找出表现最好的模型
best_model_name = max(results, key=lambda x: results[x]["cv_mean_score"])
print(f"\n最好的模型是: {best_model_name}，其平均交叉验证得分为: {results[best_model_name]['cv_mean_score']:.4f}")
