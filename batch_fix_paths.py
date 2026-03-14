"""
批量修复绝对路径
"""

import re
from pathlib import Path

def batch_fix_absolute_paths(root_dir):
    print("=" * 70)
    print("批量修复绝对路径")
    print("=" * 70)
    
    # 需要修复的文件列表
    files_to_fix = [
        "001_machine_learning_机器学习/01_supervised_learning_监督学习/02_regression_回归/04_support_vector_regression_支持向量回归/svr_demo.py",
        "001_machine_learning_机器学习/01_supervised_learning_监督学习/02_regression_回归/06_xgboost_lightgbm_catboost_regression_XGBoost_LightGBM_CatBoost回归/ensemble_regression_demo.py",
        "001_machine_learning_机器学习/01_supervised_learning_监督学习/02_regression_回归/07_neural_network_regression_神经网络回归/nn_regression_demo.py",
        "001_machine_learning_机器学习/01_supervised_learning_监督学习/02_regression_回归/08_regression_comparison_回归模型综合对比/regression_comparison_demo.py",
    ]
    
    modified_count = 0
    
    for file_rel_path in files_to_fix:
        file_path = Path(root_dir) / file_rel_path
        if file_path.exists():
            try:
                content = file_path.read_text(encoding='utf-8')
                original_content = content
                
                # 替换所有包含 "001_machine_learning_机器学习/.../images/" 的路径
                # 匹配：plt.savefig('001_machine_learning_机器学习/.../images/xxx.png'
                def replace_path(match):
                    full_path = match.group(1)
                    # 提取 images/ 后面的文件名
                    if '/images/' in full_path:
                        filename = full_path.split('/images/')[-1]
                        return f"plt.savefig('images/{filename}'"
                    return match.group(0)
                
                # 替换单引号
                content = re.sub(r"plt\.savefig\('([^']+)'", replace_path, content)
                # 替换双引号
                content = re.sub(r'plt\.savefig\("([^"]+)"', replace_path, content)
                
                if content != original_content:
                    file_path.write_text(content, encoding='utf-8')
                    print(f"✅ 修复: {file_rel_path}")
                    modified_count += 1
            except Exception as e:
                print(f"❌ 失败 {file_rel_path}: {e}")
    
    print(f"\n共修复 {modified_count} 个文件\n")

if __name__ == "__main__":
    root_dir = Path(__file__).parent
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "批量修复绝对路径" + " " * 38 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    batch_fix_absolute_paths(root_dir)
