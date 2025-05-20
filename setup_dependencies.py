#!/usr/bin/env python3
"""
安装图像处理所需的依赖项
"""
import os
import subprocess
import sys


def install_dependencies():
    """安装必要的Python依赖包"""
    print("正在安装必要的依赖包...")
    
    # 必需的依赖包
    required_packages = [
        'numpy',
        'opencv-python',
        'pillow',
        'matplotlib',
        'scikit-learn',  # 用于聚类分析
    ]
    
    # 使用pip安装
    for package in required_packages:
        print(f"安装 {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 安装成功")
        except subprocess.CalledProcessError:
            print(f"✗ {package} 安装失败")
    
    print("\n所有依赖包安装完成。")

if __name__ == "__main__":
    install_dependencies()
