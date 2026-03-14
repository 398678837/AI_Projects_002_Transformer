"""
批量修复脚本
1. 为所有demo文件所在目录创建images文件夹
2. 修复图片保存路径（统一保存到images/目录）
3. 将所有图表中文标签改为英文标签
"""

import os
import re
from pathlib import Path

# 中文到英文的映射
CHINESE_TO_ENGLISH = {
    # 标题
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
    
    # 轴标签
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
    
    # 图例和其他
    '训练损失': 'Training Loss',
    '预测结果': 'Predictions',
    '准确率': 'Accuracy',
    '精确率': 'Precision',
    '召回率': 'Recall',
    'F1分数': 'F1 Score',
    '真实标签': 'True Labels',
    '聚类标签': 'Cluster Labels',
}

def create_images_folders(root_dir):
    """为所有demo文件所在目录创建images文件夹"""
    print("=" * 70)
    print("Step 1: 创建images文件夹")
    print("=" * 70)
    
    demo_files = list(Path(root_dir).rglob("*_demo.py"))
    processed_dirs = set()
    
    for demo_file in demo_files:
        demo_dir = demo_file.parent
        images_dir = demo_dir / "images"
        
        if demo_dir not in processed_dirs:
            if not images_dir.exists():
                images_dir.mkdir(parents=True, exist_ok=True)
                print(f"✅ 创建: {images_dir}")
            processed_dirs.add(demo_dir)
    
    print(f"\n共处理 {len(processed_dirs)} 个目录\n")

def fix_savefig_paths_and_labels(root_dir):
    """修复plt.savefig路径并替换中文标签"""
    print("=" * 70)
    print("Step 2: 修复图片保存路径和中文标签")
    print("=" * 70)
    
    demo_files = list(Path(root_dir).rglob("*_demo.py"))
    modified_files = 0
    
    for demo_file in demo_files:
        try:
            content = demo_file.read_text(encoding='utf-8')
            original_content = content
            
            # 1. 修复plt.savefig路径
            # 匹配 plt.savefig('filename.png') 或 plt.savefig("filename.png")
            def replace_savefig(match):
                filename = match.group(1) or match.group(2)
                if not filename.startswith('images/') and not filename.startswith("images/"):
                    return f"plt.savefig('images/{filename}')"
                return match.group(0)
            
            content = re.sub(r"plt\.savefig\(['\"]([^'\"]+)['\"]\)", replace_savefig, content)
            
            # 2. 替换中文标签
            for chinese, english in CHINESE_TO_ENGLISH.items():
                if chinese in content:
                    content = content.replace(chinese, english)
            
            # 3. 如果有修改，写回文件
            if content != original_content:
                demo_file.write_text(content, encoding='utf-8')
                print(f"✅ 修改: {demo_file}")
                modified_files += 1
        except Exception as e:
            print(f"❌ 处理失败 {demo_file}: {e}")
    
    print(f"\n共修改 {modified_files} 个文件\n")

def check_files(root_dir):
    """检查文件状态"""
    print("=" * 70)
    print("Step 3: 检查文件状态")
    print("=" * 70)
    
    demo_files = list(Path(root_dir).rglob("*_demo.py"))
    has_images_dir = 0
    has_chinese = 0
    
    for demo_file in demo_files:
        demo_dir = demo_file.parent
        images_dir = demo_dir / "images"
        
        if images_dir.exists():
            has_images_dir += 1
        
        try:
            content = demo_file.read_text(encoding='utf-8')
            # 检查是否还有中文字符（只检查plt相关的）
            if re.search(r'plt\.(title|xlabel|ylabel|legend).*[\u4e00-\u9fff]', content):
                has_chinese += 1
                print(f"⚠️  仍有中文: {demo_file}")
        except Exception as e:
            print(f"❌ 检查失败 {demo_file}: {e}")
    
    print(f"\n检查完成:")
    print(f"  有images文件夹的目录: {has_images_dir}")
    print(f"  仍有中文标签的文件: {has_chinese}")

if __name__ == "__main__":
    root_dir = Path(__file__).parent
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "批量修复脚本" + " " * 37 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    create_images_folders(root_dir)
    fix_savefig_paths_and_labels(root_dir)
    check_files(root_dir)
    
    print("\n")
    print("✅ 修复完成！")
    print("\n")
