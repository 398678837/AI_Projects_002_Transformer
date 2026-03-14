"""
精确的图表中文标签替换脚本
只替换 plt.title, plt.xlabel, plt.ylabel, plt.legend 中的中文
"""

import os
import re
from pathlib import Path

# 中文到英文的映射（图表标签专用）
CHART_LABELS = {
    '原始数据（真实标签）': 'Original Data (True Labels)',
    'K-Means聚类结果': 'K-Means Clustering Results',
    '聚类结果': 'Clustering Results',
    '肘部法则': 'Elbow Method',
    '轮廓系数': 'Silhouette Score',
    '训练损失曲线': 'Training Loss Curve',
    'XOR问题预测结果': 'XOR Problem Predictions',
    '训练损失': 'Training Loss',
    '预测概率': 'Prediction Probability',
    '决策边界': 'Decision Boundary',
    '特征重要性': 'Feature Importance',
    '混淆矩阵': 'Confusion Matrix',
    '模型性能对比': 'Model Performance Comparison',
    'Model Performance Radar': 'Model Performance Radar',
    '决策树可视化': 'Decision Tree Visualization',
    '模型准确率': 'Model Accuracy',
    '真实值 vs 预测值': 'True Values vs Predictions',
    '残差分布': 'Residual Distribution',
    '预测结果': 'Prediction Results',
    '训练集': 'Training Set',
    '测试集': 'Test Set',
    '回归模型对比': 'Regression Model Comparison',
    'MDS降维结果': 'MDS Dimensionality Reduction Results',
    'LDA降维结果': 'LDA Dimensionality Reduction Results',
    't-SNE降维结果': 't-SNE Dimensionality Reduction Results',
    'PCA降维结果': 'PCA Dimensionality Reduction Results',
    '主成分分析': 'Principal Component Analysis',
    '解释方差比': 'Explained Variance Ratio',
    '累积解释方差': 'Cumulative Explained Variance',
    '样本': 'Sample',
    '激活函数对比': 'Activation Functions Comparison',
    'Sigmoid': 'Sigmoid',
    'Tanh': 'Tanh',
    'ReLU': 'ReLU',
    'Leaky ReLU': 'Leaky ReLU',
    '优化器对比': 'Optimizer Comparison',
    '卷积操作': 'Convolution Operation',
    '最大池化': 'Max Pooling',
    '平均池化': 'Average Pooling',
    'RNN梯度消失': 'RNN Gradient Vanishing',
    'LSTM门控': 'LSTM Gates',
    'GRU门控': 'GRU Gates',
    '序列预测': 'Sequence Prediction',
    'Transformer vs RNN': 'Transformer vs RNN',
    '训练时间': 'Training Time',
    '自注意力热图': 'Self-Attention Heatmap',
    '多头注意力': 'Multi-Head Attention',
    '位置编码': 'Positional Encoding',
    '交叉注意力': 'Cross-Attention',
    '随机森林Feature Importance': 'Random Forest Feature Importance',
    '随机森林特征重要性': 'Random Forest Feature Importance',
    'SVM决策边界': 'SVM Decision Boundary',
    '线性回归': 'Linear Regression',
    '多项式回归': 'Polynomial Regression',
    '岭回归': 'Ridge Regression',
    'LASSO回归': 'LASSO Regression',
    '弹性网络': 'Elastic Net',
    '支持向量回归': 'Support Vector Regression',
    '树回归': 'Tree Regression',
    'XGBoost回归': 'XGBoost Regression',
    'LightGBM回归': 'LightGBM Regression',
    'CatBoost回归': 'CatBoost Regression',
    '神经网络回归': 'Neural Network Regression',
    '聚类数量 k': 'Number of Clusters k',
    '惯性（Inertia）': 'Inertia',
    'PCA 1': 'PCA 1',
    'PCA 2': 'PCA 2',
    '类别': 'Class',
    '聚类': 'Cluster',
    'Epoch': 'Epoch',
    'Loss': 'Loss',
    '输入': 'Input',
    '特征': 'Feature',
    '重要性': 'Importance',
    '真实值': 'True Values',
    '预测值': 'Predictions',
    '残差': 'Residuals',
    '样本数': 'Number of Samples',
    '频率': 'Frequency',
    '主成分': 'Principal Component',
    'x': 'x',
    'f(x)': 'f(x)',
    'Step': 'Step',
    '输入特征': 'Input Feature',
    '输出特征': 'Output Feature',
    '池化后': 'After Pooling',
    '时间步': 'Time Step',
    '隐藏状态': 'Hidden State',
    '门控值': 'Gate Value',
    '位置': 'Position',
    '编码值': 'Encoding Value',
    '训练损失': 'Training Loss',
    '预测结果': 'Predictions',
    '准确率': 'Accuracy',
    '精确率': 'Precision',
    '召回率': 'Recall',
    'F1分数': 'F1 Score',
    '真实标签': 'True Labels',
    '聚类标签': 'Cluster Labels',
    '树的数量': 'Number of Trees',
    '每棵树的最大深度': 'Max Depth per Tree',
    '样本特征': 'Sample Features',
    '实际Class': 'Actual Class',
    '预测Class': 'Predicted Class',
    'PCA降维到2维结果': 'PCA Dimensionality Reduction to 2D Results',
    '主成分 1': 'Principal Component 1',
    '主成分 2': 'Principal Component 2',
    '方差)': 'variance)',
    'Principal Component数量': 'Number of Principal Components',
    'Explained Variance Ratio率': 'Explained Variance Ratio',
    'PCA解释方差分析': 'PCA Explained Variance Analysis',
    '原始标签': 'Original Labels',
    'Principal Component 1': 'Principal Component 1',
    'Principal Component 2': 'Principal Component 2',
    '集成树Model Performance Comparison': 'Ensemble Trees Model Performance Comparison',
    'sepal length (标准化)': 'sepal length (normalized)',
    'sepal width (标准化)': 'sepal width (normalized)',
    'SVMDecision Boundary (RBF核)': 'SVM Decision Boundary (RBF kernel)',
    '实际房价': 'Actual House Price',
    '预测房价': 'Predicted House Price',
    'Linear Regression：实际房价 vs 预测房价': 'Linear Regression: Actual vs Predicted House Price',
    'Linear Regression：Residuals分析': 'Linear Regression: Residuals Analysis',
    '收入中位数 (MedInc)': 'Median Income (MedInc)',
    '房价中位数': 'Median House Price',
    'Importance（系数绝对值）': 'Importance (Absolute Coefficient Value)',
    'alpha（正则化参数）': 'alpha (Regularization Parameter)',
    'R²评分': 'R² Score',
    '不同alpha值对模型性能的影响': 'Effect of Different alpha Values on Model Performance',
    '树的最大深度': 'Max Tree Depth',
    '不同树深度对模型性能的影响': 'Effect of Different Tree Depths on Model Performance',
    '不同树数量对随机森林性能的影响': 'Effect of Different Tree Counts on Random Forest Performance',
    '模型': 'Model',
    '模型R² Score对比': 'Model R² Score Comparison',
    '模型MSE对比': 'Model MSE Comparison',
    '迭代次数': 'Iteration',
    '损失': 'Loss',
    '激活函数': 'Activation Function',
    '不同激活函数的R² Score对比': 'R² Score Comparison for Different Activation Functions',
    'Test SetR² Score排序': 'Test Set R² Score Ranking',
    '层次Clustering Results': 'Hierarchical Clustering Results',
    '高斯混合模型结果': 'Gaussian Mixture Model Results',
    '层次Cluster树状图': 'Hierarchical Cluster Dendrogram',
    'Sample索引': 'Sample Index',
    '距离': 'Distance',
    't-SNE降维 - MNIST数据集': 't-SNE Dimensionality Reduction - MNIST Dataset',
    'LDA降维 - 鸢尾花数据集': 'LDA Dimensionality Reduction - Iris Dataset',
    'MDS降维 - 鸢尾花数据集': 'MDS Dimensionality Reduction - Iris Dataset',
    '网格世界 - MDP示例': 'Grid World - MDP Example',
    '列': 'Column',
    '行': 'Row',
    'ModelR² Score对比': 'Model R² Score Comparison',
    'ModelMSE对比': 'Model MSE Comparison',
    '不同Activation Function的R² Score对比': 'R² Score Comparison for Different Activation Functions',
    '高斯混合Model结果': 'Gaussian Mixture Model Results',
}

def replace_chart_labels(root_dir):
    """只替换图表标签中的中文"""
    print("=" * 70)
    print("替换图表标签中的中文")
    print("=" * 70)
    
    demo_files = list(Path(root_dir).rglob("*_demo.py"))
    modified_files = 0
    
    for demo_file in demo_files:
        try:
            content = demo_file.read_text(encoding='utf-8')
            original_content = content
            
            # 只匹配 plt.title, plt.xlabel, plt.ylabel, plt.legend
            # 模式: plt.function('中文内容') 或 plt.function("中文内容")
            
            def replace_in_chart_function(match):
                func = match.group(1)
                content = match.group(2) or match.group(3)
                
                # 替换内容中的中文
                new_content = content
                for chinese, english in CHART_LABELS.items():
                    new_content = new_content.replace(chinese, english)
                
                if new_content != content:
                    return f"{func}('{new_content}')"
                return match.group(0)
            
            # 匹配 plt.title('...'), plt.xlabel('...'), plt.ylabel('...'), plt.legend('...')
            # 支持单引号和双引号
            pattern = r'(plt\.(?:title|xlabel|ylabel|legend))\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
            content = re.sub(pattern, replace_in_chart_function, content)
            
            # 如果有修改，写回文件
            if content != original_content:
                demo_file.write_text(content, encoding='utf-8')
                print(f"✅ 修改: {demo_file}")
                modified_files += 1
        except Exception as e:
            print(f"❌ 处理失败 {demo_file}: {e}")
    
    print(f"\n共修改 {modified_files} 个文件\n")

if __name__ == "__main__":
    root_dir = Path(__file__).parent
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "图表中文标签替换脚本" + " " * 37 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    replace_chart_labels(root_dir)
    
    print("\n")
    print("✅ 替换完成！")
    print("\n")
