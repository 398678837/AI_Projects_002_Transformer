"""
找出所有还有中文图表标签的文件
"""

import re
from pathlib import Path

def find_chinese_chart_labels(root_dir):
    print("=" * 70)
    print("查找所有包含中文图表标签的demo文件")
    print("=" * 70)
    
    demo_files = list(Path(root_dir).rglob("*_demo.py"))
    files_with_chinese = []
    
    for demo_file in demo_files:
        try:
            content = demo_file.read_text(encoding='utf-8')
            
            # 查找包含中文的 plt.title, plt.xlabel, plt.ylabel, plt.legend
            pattern = r'plt\.(?:title|xlabel|ylabel|legend)\s*\(\s*[\'"]([^\'"]*[\u4e00-\u9fff][^\'"]*)[\'"]\s*\)'
            matches = re.findall(pattern, content)
            
            if matches:
                files_with_chinese.append((demo_file, matches))
                print(f"\n📄 {demo_file}")
                for match in matches:
                    print(f"  🔤 {match}")
        except Exception as e:
            print(f"❌ 检查失败 {demo_file}: {e}")
    
    print(f"\n共找到 {len(files_with_chinese)} 个文件还有中文图表标签\n")
    return files_with_chinese

if __name__ == "__main__":
    root_dir = Path(__file__).parent
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "查找中文图表标签脚本" + " " * 40 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    find_chinese_chart_labels(root_dir)
