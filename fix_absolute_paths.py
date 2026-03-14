"""
修复绝对路径的plt.savefig
"""

import re
from pathlib import Path

def fix_absolute_paths(root_dir):
    print("=" * 70)
    print("修复绝对路径的plt.savefig")
    print("=" * 70)
    
    demo_files = list(Path(root_dir).rglob("*_demo.py"))
    modified_files = 0
    
    for demo_file in demo_files:
        try:
            content = demo_file.read_text(encoding='utf-8')
            original_content = content
            
            # 匹配绝对路径的plt.savefig
            # 格式：001_machine_learning_机器学习/.../images/filename.png
            def fix_path(match):
                full_path = match.group(1) or match.group(2)
                # 提取文件名部分（最后一个/images/后面的部分）
                if '/images/' in full_path:
                    filename = full_path.split('/images/')[-1]
                    return f"plt.savefig('images/{filename}')"
                return match.group(0)
            
            pattern = r'plt\.savefig\s*\(\s*[\'"]([^\'"]+[\'"])'
            content = re.sub(r'plt\.savefig\s*\(\s*[\'"]([^\'"]+)[\'"]', fix_path, content)
            
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
    print("║" + " " * 10 + "修复绝对路径脚本" + " " * 42 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    fix_absolute_paths(root_dir)
