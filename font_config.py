# -*- coding: utf-8 -*-
"""
中文字体配置模块
解决Matplotlib中文显示乱码问题
"""

import matplotlib.pyplot as plt
import matplotlib
import os
import sys

# 可用的中文字体列表（按优先级排序）
CHINESE_FONTS = [
    # Windows 常用字体
    'Microsoft YaHei',
    'SimHei',
    'SimSun',
    'FangSong',
    'KaiTi',
    'STSong',
    # Linux 常用字体
    'WenQuanYi Micro Hei',
    'WenQuanYi Zen Hei',
    'Noto Sans CJK SC',
    'Noto Sans CJK TC',
    # Mac 常用字体
    'PingFang SC',
    'Heiti SC',
    'STHeiti',
    # 回退字体
    'DejaVu Sans'
]

def setup_chinese_font():
    """
    设置中文字体，解决乱码问题
    """
    # 方案1: 尝试从系统字体中找到可用的中文字体
    from matplotlib.font_manager import FontManager
    
    fm = FontManager()
    available_fonts = set(f.name for f in fm.ttflist)
    
    # 查找可用的中文字体
    available_chinese_fonts = []
    for font in CHINESE_FONTS:
        if font in available_fonts:
            available_chinese_fonts.append(font)
    
    if available_chinese_fonts:
        plt.rcParams['font.sans-serif'] = [available_chinese_fonts[0]] + CHINESE_FONTS
        print(f"✅ 找到可用的中文字体: {available_chinese_fonts[0]}")
    else:
        print("⚠️ 未找到系统自带的中文字体")
        print("   建议下载安装: https://github.com/adobe-fonts/source-han-sans/releases")
        print("   或使用: pip install matplotlib-font")
    
    # 修复负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    return available_chinese_fonts[0] if available_chinese_fonts else None


def use_english_labels():
    """
    使用英文标签，完全避免中文问题
    """
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = True


def download_noto_font():
    """
    下载思源中文字体（需要网络）
    """
    try:
        import urllib.request
        import zipfile
        import shutil
        
        font_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        os.makedirs(font_dir, exist_ok=True)
        
        font_path = os.path.join(font_dir, 'NotoSansCJK-Regular.ttc')
        
        if os.path.exists(font_path):
            print(f"字体已存在于: {font_path}")
            return font_path
        
        print("正在下载思源中文字体...")
        url = "https://github.com/googlefonts/noto-cjk/releases/download/Sans2.004/03_NotoSansCJK-OTC.zip"
        
        # 下载字体文件
        zip_path = os.path.join(font_dir, 'noto-font.zip')
        urllib.request.urlretrieve(url, zip_path)
        
        # 解压
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if 'NotoSansCJK-Regular.ttc' in file:
                    zip_ref.extract(file, font_dir)
                    break
        
        # 移动文件
        for root, dirs, files in os.walk(font_dir):
            for f in files:
                if 'NotoSansCJK-Regular.ttc' in f:
                    src = os.path.join(root, f)
                    shutil.move(src, font_path)
                    break
        
        os.remove(zip_path)
        
        # 添加字体到matplotlib
        matplotlib.font_manager.fontManager.addfont(font_path)
        
        print(f"✅ 字体已下载到: {font_path}")
        return font_path
        
    except Exception as e:
        print(f"❌ 字体下载失败: {e}")
        return None


def get_font_config():
    """
    获取推荐的字体配置
    """
    return {
        'font.sans-serif': CHINESE_FONTS,
        'axes.unicode_minus': False,
        'figure.figsize': (10, 6),
        'font.size': 12
    }


# 自动初始化
if __name__ == '__main__':
    print("=" * 50)
    print("Matplotlib 中文字体配置")
    print("=" * 50)
    
    font = setup_chinese_font()
    
    print("\n当前配置:")
    print(f"  字体: {plt.rcParams['font.sans-serif'][0]}")
    print(f"  负号: {plt.rcParams['axes.unicode_minus']}")
    
    print("\n" + "=" * 50)
