"""一键云端训练：自动同步代码/数据到云端 -> 远程触发训练（实时流式查看日志） -> 训练完成后自动拉回模型权重。

配置依赖 scripts/Tools/cloud_config.py（见 cloud_config.example.py）。

用法示例：
    # 一键跑 SFT，并在完成后自动拉回 models/ 目录下的产出
    python scripts/Tools/cloud_train.py --stage full_sft --batch_size 224 --use_compile 1

    # 指定云端 python 解析器（如 conda 环境）
    python scripts/Tools/cloud_train.py --stage pretrain --py_bin ~/miniforge3/envs/minimind/bin/python

    # 只触发训练与同步，训练完不自动 pull
    python scripts/Tools/cloud_train.py --stage lora --no-pull
"""
import os
import shlex
import shutil
import subprocess
import sys

# 确保能正常导入项目内部模块
sys.path.insert(0, os.getcwd())

from scripts.Tools.sync_data import load_config, run_rsync, build_ssh, DEFAULT_DIRS


def build_ssh_cmd(cfg, remote_command):
    """构造远程 SSH 执行命令。支持 sshpass。"""
    ssh_exec = ['ssh', '-p', str(cfg['port']), f"{cfg['user']}@{cfg['host']}", remote_command]
    if cfg['password']:
        if shutil.which('sshpass') is None:
            sys.exit('配置了密码但未安装 sshpass：请 `apt install sshpass`，或改用 SSH 密钥免密（推荐）')
        ssh_exec = ['sshpass', '-p', cfg['password']] + ssh_exec
    return ssh_exec


def main():
    cfg = load_config()
    if not cfg['host']:
        sys.exit('错误：未配置云端主机。请在 scripts/Tools/cloud_config.py 设置 HOST（参考 cloud_config.example.py）')

    # 解析可透传给脚本的剩余参数
    # 如果用户传了 --py_bin / --no-push / --no-pull，单独提取
    raw_args = sys.argv[1:]
    
    no_push = '--no-push' in raw_args
    if no_push:
        raw_args.remove('--no-push')

    no_pull = '--no-pull' in raw_args
    if no_pull:
        raw_args.remove('--no-pull')

    py_bin = 'python'
    if '--py_bin' in raw_args:
        idx = raw_args.index('--py_bin')
        py_bin = raw_args[idx + 1]
        del raw_args[idx:idx + 2]

    target_script = 'scripts/Trainer/train.py'
    if '--script' in raw_args:
        idx = raw_args.index('--script')
        target_script = raw_args[idx + 1]
        del raw_args[idx:idx + 2]

    # 1. 自动同步本地数据/代码/权重到云端
    ssh_env = build_ssh(cfg['port'], cfg['password'])
    dst_prefix = f"{cfg['user']}@{cfg['host']}:{cfg['path']}"
    root = os.getcwd()

    if not no_push:
        print("=== Step 1/3: 推送本地数据与权重至云端 ===")
        for d in DEFAULT_DIRS:
            local_dir = os.path.join(root, d)
            remote_dir = f"{dst_prefix}/{d}"
            if os.path.exists(local_dir):
                run_rsync('push', local_dir, remote_dir, ssh_env, dry_run=False)

    # 2. 构建远程命令并启动训练
    print("\n=== Step 2/3: 在云端启动训练（实时输出日志）===")
    train_args_str = shlex.join(raw_args)
    remote_cmd = f"cd {shlex.quote(cfg['path'])} && {py_bin} {target_script} {train_args_str}"
    full_cmd = build_ssh_cmd(cfg, remote_cmd)
    
    print(f"> 远程命令: {remote_cmd}\n")
    try:
        # 使用 pty/pipe 保持实时输出
        res = subprocess.run(full_cmd)
        if res.returncode != 0:
            sys.exit(f"\n错误：云端训练异常终止，退出码: {res.returncode}")
    except KeyboardInterrupt:
        sys.exit("\n本地已取消监控。云端进程可能仍在使用 GPU，请检查远端后台。")

    print("\n=== Step 3/3: 训练完成，自动从云端拉回最新权重 (models/) ===")
    if not no_pull:
        local_models = os.path.join(root, 'models')
        remote_models = f"{dst_prefix}/models"
        run_rsync('pull', local_models, remote_models, ssh_env, dry_run=False)
        print("🎉 一键云端训练全流程完成！新模型权重已存入本地 models/ 目录。")


if __name__ == '__main__':
    main()
