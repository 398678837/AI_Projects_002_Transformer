"""
检查图片保存路径
"""

import re
from pathlib import Path

def check_savefig_paths(root_dir):
    print("=" * 70)
    print("检查所有demo文件的plt.savefig路径")
    print("=" * 70)
    
    demo_files = list(Path(root_dir).rglob("*_demo.py"))
    bad_paths = []
    
    for demo_file in demo_files:
        try:
            content = demo_file.read_text(encoding='utf-8')
            
            # 查找所有 plt.savefig
            pattern = r'plt\.savefig\s*\(\s*[\'"]([^\'"]+)[\'"]'
            matches = re.findall(pattern, content)
            
            for match in matches:
                if not match.startswith('images/'):
                    bad_paths.append((demo_file, match))
                    print(f"❌ {demo_file}: {match}")
        except Exception as e:
            print(f"❌ 检查失败 {demo_file}: {e}")
    
    if not bad_paths:
        print("✅ 所有图片路径都正确！")
    else:
        print(f"\n共发现 {len(bad_paths)} 个问题路径")
    
    return bad_paths

if __name__ == "__main__":
    root_dir = Path(__file__).parent
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "图片路径检查脚本" + " " * 38 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    check_savefig_paths(root_dir)
