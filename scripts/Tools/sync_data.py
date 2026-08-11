"""本地 ↔ 云 GPU 机数据同步（基于 rsync over SSH）。

代码走 git，数据/权重/续训档走 rsync——两者解耦，避免把几十 GB 数据塞进 git。
远端只需有 SSH 可达（云服务器 / 内网机器均可）。

用法（在项目根目录执行）：
    python scripts/Tools/sync_data.py pull              # 云端 → 本地（拉数据/权重）
    python scripts/Tools/sync_data.py push              # 本地 → 云端（传数据/权重）
    python scripts/Tools/sync_data.py pull --dirs resource models   # 只同步部分目录
    python scripts/Tools/sync_data.py pull --dry-run                # 预览不动手

环境变量配置云端目标：
    CLOUD_HOST   必填，SSH 主机（IP 或域名）
    CLOUD_USER   SSH 用户（默认 $USER）
    CLOUD_PORT   SSH 端口（默认 22）
    CLOUD_PATH   云端项目路径（默认 ~/minimind）
"""
import argparse
import os
import shlex
import subprocess
import sys

# 默认同步的目录（相对项目根；checkpoints 为续训档）
DEFAULT_DIRS = ['resource', 'models', 'checkpoints']


def run_rsync(direction, local_dir, remote_dir, ssh, dry_run):
    cmd = ['rsync', '-avz', '--partial', '--progress', '-e', ssh,
           '--exclude', '__pycache__', '--exclude', '*.tmp']
    if dry_run:
        cmd.append('-n')
    if direction == 'pull':
        cmd += [f'{remote_dir}/', f'{local_dir}/']
    else:
        cmd += [f'{local_dir}/', f'{remote_dir}/']
    print('>', shlex.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='本地 ↔ 云 GPU 机数据同步（rsync over SSH）')
    parser.add_argument('direction', choices=['pull', 'push'],
                        help='pull = 云端 → 本地；push = 本地 → 云端')
    parser.add_argument('--dirs', nargs='*', default=DEFAULT_DIRS,
                        help=f'要同步的目录（默认: {" ".join(DEFAULT_DIRS)}）')
    parser.add_argument('--dry-run', action='store_true', help='只预览不传输（rsync -n）')
    args = parser.parse_args()

    host = os.environ.get('CLOUD_HOST')
    if not host:
        sys.exit('错误：请设置环境变量 CLOUD_HOST（云端 SSH 主机），如 export CLOUD_HOST=1.2.3.4')
    user = os.environ.get('CLOUD_USER', os.environ.get('USER', 'root'))
    port = os.environ.get('CLOUD_PORT', '22')
    cloud_path = os.environ.get('CLOUD_PATH', '~/minimind')

    root = os.getcwd()
    ssh = f'ssh -p {port}'
    dst = f'{user}@{host}'

    for d in args.dirs:
        local_dir = os.path.join(root, d)
        remote_dir = f'{dst}:{cloud_path}/{d}'
        run_rsync(args.direction, local_dir, remote_dir, ssh, args.dry_run)

    print(f'同步完成（{args.direction}）')


if __name__ == '__main__':
    main()
