"""
修复转义字符问题
"""

import re
from pathlib import Path

def fix_escape_chars(root_dir):
    print("=" * 70)
    print("修复转义字符问题")
    print("=" * 70)
    
    rl_dir = Path(root_dir) / "001_machine_learning_机器学习" / "03_reinforcement_learning_强化学习"
    demo_files = list(rl_dir.rglob("*_demo.py"))
    
    modified_count = 0
    
    for demo_file in demo_files:
        try:
            content = demo_file.read_text(encoding='utf-8')
            original_content = content
            
            # 修复 \'images\' 为 'images'
            content = content.replace("\\'images\\'", "'images'")
            
            if content != original_content:
                demo_file.write_text(content, encoding='utf-8')
                print(f"✅ 修复: {demo_file.name}")
                modified_count += 1
        except Exception as e:
            print(f"❌ 失败 {demo_file}: {e}")
    
    print(f"\n共修复 {modified_count} 个文件\n")

if __name__ == "__main__":
    import sys
    root_dir = Path(sys.argv[0]).parent
    fix_escape_chars(root_dir)
