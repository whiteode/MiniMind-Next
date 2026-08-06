"""pytest 公共配置：把项目根目录加入 sys.path，使用例可 import scripts.Deploy.*。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
