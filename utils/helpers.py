"""
AI学习路径项目的工具函数
"""

import os
import yaml
from pathlib import Path

def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.parent

def load_config(config_path="config/config.yaml"):
    """从YAML文件加载配置"""
    config_file = get_project_root() / config_path
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_dir(directory):
    """确保目录存在，不存在则创建"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_data_path(data_type="raw", filename=None):
    """获取数据目录路径"""
    config = load_config()
    data_dir = get_project_root() / config['paths']['data'][data_type]
    if filename:
        return data_dir / filename
    return data_dir

def print_progress(current, total, description=""):
    """打印进度条"""
    percent = (current / total) * 100
    bar_length = 50
    filled = int(bar_length * current // total)
    bar = '█' * filled + '-' * (bar_length - filled)
    print(f'\r{description} |{bar}| {percent:.1f}%', end='', flush=True)
    if current == total:
        print()
