"""
批量修复强化学习demo的图片保存路径
"""

import os
import re
from pathlib import Path

def add_script_dir_fix(root_dir):
    print("=" * 70)
    print("批量修复强化学习demo")
    print("=" * 70)
    
    # 获取所有demo文件
    rl_dir = Path(root_dir) / "001_machine_learning_机器学习" / "03_reinforcement_learning_强化学习"
    demo_files = list(rl_dir.rglob("*_demo.py"))
    
    modified_count = 0
    
    for demo_file in demo_files:
        try:
            content = demo_file.read_text(encoding='utf-8')
            original_content = content
            
            # 检查是否需要添加script_dir逻辑
            needs_script_dir = False
            
            # 1. 如果没有 import os 且使用了 plt.savefig
            if 'import os' not in content and "plt.savefig" in content:
                needs_script_dir = True
            
            if needs_script_dir and 'script_dir' not in content:
                # 在import语句后添加script_dir
                # 找到 import matplotlib 行
                import_pattern = r'(import matplotlib\.pyplot as plt)'
                if re.search(import_pattern, content):
                    content = re.sub(
                        import_pattern,
                        r'\1\n\nscript_dir = os.path.dirname(os.path.abspath(__file__))\nimages_dir = os.path.join(script_dir, \'images\')',
                        content
                    )
            
            # 2. 修复 plt.savefig('images/xxx') 为 plt.savefig(os.path.join(images_dir, 'xxx'))
            if 'script_dir' in content:
                def fix_savefig(match):
                    # 提取文件名
                    path_content = match.group(1)
                    if 'images/' in path_content:
                        filename = path_content.split('images/')[-1].strip("'\")")
                        return f"plt.savefig(os.path.join(images_dir, '{filename}'))"
                    return match.group(0)
                
                # 匹配 plt.savefig('images/xxx') 或 plt.savefig("images/xxx")
                content = re.sub(r"plt\.savefig\(['\"]images/([^'\"]+)['\"]\)", fix_savefig, content)
            
            if content != original_content:
                demo_file.write_text(content, encoding='utf-8')
                print(f"✅ 修复: {demo_file.name}")
                modified_count += 1
        except Exception as e:
            print(f"❌ 失败 {demo_file}: {e}")
    
    print(f"\n共修复 {modified_count} 个文件\n")

if __name__ == "__main__":
    root_dir = Path(__file__).parent
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "批量修复强化学习demo" + " " * 40 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    add_script_dir_fix(root_dir)
