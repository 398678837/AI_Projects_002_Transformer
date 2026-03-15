"""
批量添加 import os 到强化学习demo
"""

import re
from pathlib import Path

def add_import_os(root_dir):
    print("=" * 70)
    print("批量添加 import os")
    print("=" * 70)
    
    rl_dir = Path(root_dir) / "001_machine_learning_机器学习" / "03_reinforcement_learning_强化学习"
    demo_files = list(rl_dir.rglob("*_demo.py"))
    
    modified_count = 0
    
    for demo_file in demo_files:
        try:
            content = demo_file.read_text(encoding='utf-8')
            original_content = content
            
            # 如果使用了 script_dir 但没有 import os
            if 'script_dir' in content and 'import os' not in content:
                # 在第一个 import 后添加 import os
                # 找到 import numpy 或其他 import
                import_match = re.search(r'^(import .+)$', content, re.MULTILINE)
                if import_match:
                    # 在第一个 import 后插入 import os
                    content = content.replace(
                        import_match.group(1),
                        import_match.group(1) + '\nimport os'
                    )
            
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
    add_import_os(root_dir)
