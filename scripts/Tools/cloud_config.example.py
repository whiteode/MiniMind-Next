"""云端目标配置模板：复制为 cloud_config.py 并填写真实值（cloud_config.py 已 gitignore）。"""

HOST = '1.2.3.4'          # 必填：云端 SSH 主机（IP 或域名）
USER = 'xavier'           # SSH 用户（留空 = 当前 $USER）
PORT = 22                 # SSH 端口
PATH = '~/minimind'       # 云端项目路径
PASSWORD = ''             # 可选：SSH 密码（非空时用 sshpass；推荐 SSH 密钥免密留空）
