"""本地 ↔ 云 GPU 机数据同步（基于 rsync over SSH）。

代码走 git，数据/权重/续训档走 rsync——两者解耦，避免把几十 GB 数据塞进 git。
云端目标统一在 scripts/Tools/cloud_config.py 配置（含可选密码，见 cloud_config.example.py）。

用法（在项目根目录执行）：
    python scripts/Tools/sync_data.py pull              # 云端 → 本地（拉数据/权重）
    python scripts/Tools/sync_data.py push              # 本地 → 云端（传数据/权重）
    python scripts/Tools/sync_data.py pull --dirs resource models   # 只同步部分目录
    python scripts/Tools/sync_data.py pull --dry-run                # 预览不动手

配置优先级：cloud_config.py > 环境变量（CLOUD_HOST/USER/PORT/PATH/PASSWORD）> 默认值
密码：配置了 PASSWORD 时用 sshpass（需已安装）；推荐优先用 SSH 密钥免密。
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys

# 默认同步的目录（相对项目根；checkpoints 为续训档）
DEFAULT_DIRS = ['resource', 'models', 'checkpoints']


def load_config():
    """从 scripts/Tools/cloud_config.py 读取配置；文件缺失或字段为空时回退到环境变量/默认值。"""
    cfg = {}
    try:
        from scripts.Tools import cloud_config as cc
        cfg = {k: getattr(cc, k, None) for k in ('HOST', 'USER', 'PORT', 'PATH', 'PASSWORD')}
    except Exception:
        pass
    host = os.environ.get('CLOUD_HOST') or cfg.get('HOST') or ''
    user = os.environ.get('CLOUD_USER') or cfg.get('USER') or os.environ.get('USER', 'root')
    port = os.environ.get('CLOUD_PORT') or cfg.get('PORT') or '22'
    path = os.environ.get('CLOUD_PATH') or cfg.get('PATH') or '~/minimind'
    password = os.environ.get('CLOUD_PASSWORD') or cfg.get('PASSWORD') or ''
    return {'host': host, 'user': user, 'port': str(port), 'path': path, 'password': password}


def build_ssh(port, password):
    """构造 rsync -e 的远端 shell 命令；配了密码用 sshpass。"""
    if password:
        if shutil.which('sshpass') is None:
            sys.exit('配置了密码但未安装 sshpass：请 `apt install sshpass`，或改用 SSH 密钥免密（推荐）')
        return f'sshpass -p {shlex.quote(password)} ssh -p {port}'
    return f'ssh -p {port}'


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

    cfg = load_config()
    if not cfg['host']:
        sys.exit('错误：未配置云端主机。请在 scripts/Tools/cloud_config.py 设置 HOST（参考 cloud_config.example.py）')

    ssh = build_ssh(cfg['port'], cfg['password'])
    dst = f'{cfg["user"]}@{cfg["host"]}'
    root = os.getcwd()

    for d in args.dirs:
        local_dir = os.path.join(root, d)
        remote_dir = f'{dst}:{cfg["path"]}/{d}'
        run_rsync(args.direction, local_dir, remote_dir, ssh, args.dry_run)

    print(f'同步完成（{args.direction}）')


if __name__ == '__main__':
    main()
