"""
精确修复绝对路径的plt.savefig
"""

import re
from pathlib import Path

def fix_absolute_paths_precise(root_dir):
    print("=" * 70)
    print("精确修复绝对路径的plt.savefig")
    print("=" * 70)
    
    demo_files = list(Path(root_dir).rglob("*_demo.py"))
    modified_files = 0
    
    for demo_file in demo_files:
        try:
            content = demo_file.read_text(encoding='utf-8')
            original_content = content
            
            # 修复：001_machine_learning_机器学习/.../images/filename.png → images/filename.png
            # 正则匹配：包含/images/的长路径
            def replace_func(match):
                full_path = match.group(1)
                # 如果路径包含 /images/，提取文件名
                if '/images/' in full_path:
                    filename = full_path.split('/images/')[-1]
                    return f"plt.savefig('images/{filename}'"
                return match.group(0)
            
            # 匹配 plt.savefig('长路径'
            content = re.sub(r"plt\.savefig\('([^']+)'", replace_func, content)
            content = re.sub(r'plt\.savefig\("([^"]+)"', replace_func, content)
            
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
    print("║" + " " * 10 + "精确修复绝对路径脚本" + " " * 37 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    fix_absolute_paths_precise(root_dir)
