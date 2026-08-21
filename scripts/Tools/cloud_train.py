"""云端训练轻量运行工具：毫秒级同步本地代码 -> 云端激活 Conda 环境 -> 执行训练并实时流式输出日志。

设计思路：
  - 数据集 (resource/) 和 模型权重 (models/) 由用户自行按需传输，不在此处浪费时间与带宽。
  - 本工具只负责快速增量同步本地代码 (scripts/) 到云端，并在云端 Conda 环境中执行具体指令，实时透传输出。

用法格式：
    python scripts/Tools/cloud_train.py run <conda_env> <command...>

示例：
    # 激活云端的 minimind 环境，运行 full_sft 训练
    python scripts/Tools/cloud_train.py run minimind python scripts/Trainer/train.py --stage full_sft --batch_size 224 --use_compile 1

    # 激活云端的 base 或 pytorch 环境运行预训练
    python scripts/Tools/cloud_train.py run pytorch python scripts/Trainer/train.py --stage pretrain --batch_size 224

    # 支持多卡 DDP 指令
    python scripts/Tools/cloud_train.py run minimind torchrun --nproc_per_node 2 scripts/Trainer/train.py --stage full_sft

    # 可选附加开关：
    #   --no-sync     跳过代码同步，直接在云端执行命令
    #   --shutdown    训练命令执行完毕后自动关机（按量计费省钱防漏）

配置依赖 scripts/Tools/cloud_config.py（见 cloud_config.example.py）。
"""
import os
import shlex
import shutil
import subprocess
import sys

# 确保能正常导入项目内部模块
sys.path.insert(0, os.getcwd())

from scripts.Tools.sync_data import load_config, run_rsync, build_ssh


def build_ssh_cmd(cfg, remote_command):
    """构造远程 SSH 执行命令。支持 sshpass。"""
    ssh_exec = ['ssh', '-p', str(cfg['port']), f"{cfg['user']}@{cfg['host']}", remote_command]
    if cfg['password']:
        if shutil.which('sshpass') is None:
            sys.exit('配置了密码但未安装 sshpass：请 `apt install sshpass`，或改用 SSH 密钥免密（推荐）')
        ssh_exec = ['sshpass', '-p', cfg['password']] + ssh_exec
    return ssh_exec


def sync_code(cfg, ssh_env):
    """仅快速同步本地代码目录 (scripts/) 到云端，排除 __pycache__ 等无关文件。"""
    root = os.getcwd()
    local_scripts = os.path.join(root, 'scripts')
    dst = f"{cfg['user']}@{cfg['host']}"
    remote_scripts = f"{dst}:{cfg['path']}/scripts"
    print("⚡ 正在增量同步本地代码 (scripts/) 到云端...")
    run_rsync('push', local_scripts, remote_scripts, ssh_env, dry_run=False)
    print("✅ 代码同步完成！\n")


def main():
    raw_args = sys.argv[1:]
    if not raw_args or raw_args[0] in ('-h', '--help'):
        print(__doc__)
        return

    subcmd = raw_args[0]
    if subcmd != 'run':
        # 如果用户直接传了 conda_env 或其他命令，提示正确用法
        sys.exit("错误：用法应为 `python scripts/Tools/cloud_train.py run <conda_env> <command...>`\n查看帮助请运行 `python scripts/Tools/cloud_train.py --help`")

    rest = raw_args[1:]
    if not rest:
        sys.exit("错误：缺少 conda 环境名称。用法：`python scripts/Tools/cloud_train.py run <conda_env> <command...>`")

    # 提取可选标记
    no_sync = '--no-sync' in rest
    if no_sync:
        rest.remove('--no-sync')

    auto_shutdown = False
    if '--shutdown' in rest:
        auto_shutdown = True
        rest.remove('--shutdown')
    elif '--auto-shutdown' in rest:
        auto_shutdown = True
        rest.remove('--auto-shutdown')

    if not rest:
        sys.exit("错误：缺少要执行的具体命令。例如：`python scripts/Tools/cloud_train.py run minimind python scripts/Trainer/train.py ...`")

    conda_env = rest[0]
    command_to_run = rest[1:]

    if not command_to_run:
        sys.exit(f"错误：请在 conda 环境 `{conda_env}` 后面跟上你要执行的具体指令。")

    cfg = load_config()
    if not cfg['host']:
        sys.exit('错误：未配置云端主机。请在 scripts/Tools/cloud_config.py 设置 HOST（参考 cloud_config.example.py）')

    ssh_env = build_ssh(cfg['port'], cfg['password'])

    # 1. 快速将本地代码增量推送到云端（仅 scripts/ 目录，毫秒级完成）
    if not no_sync:
        sync_code(cfg, ssh_env)

    # 2. 构造云端 bash 命令：进入项目目录 -> 初始化并激活 conda 环境 -> 执行用户命令
    cmd_str = shlex.join(command_to_run)
    shutdown_str = " && (sudo shutdown || shutdown || poweroff)" if auto_shutdown else ""

    # 使用 bash 登录 shell 或 source conda.sh 确保 conda activate 可用
    remote_script = (
        f"cd {shlex.quote(cfg['path'])} && "
        f"source ~/.bashrc 2>/dev/null || true; "
        f"eval \"$(conda shell.bash hook 2>/dev/null || ~/miniforge3/bin/conda shell.bash hook 2>/dev/null || ~/miniconda3/bin/conda shell.bash hook 2>/dev/null || anaconda/bin/conda shell.bash hook 2>/dev/null)\"; "
        f"conda activate {shlex.quote(conda_env)} && "
        f"{cmd_str}{shutdown_str}"
    )

    full_ssh_cmd = build_ssh_cmd(cfg, f"bash -c {shlex.quote(remote_script)}")

    print(f"🚀 [云端执行] 环境: ({conda_env}) | 命令: {cmd_str}")
    if auto_shutdown:
        print("💡 已开启 --shutdown，训练结束后远端将自动关机。")
    print("=" * 70 + "\n")

    try:
        res = subprocess.run(full_ssh_cmd)
        if res.returncode != 0:
            sys.exit(f"\n❌ 云端任务异常终止，退出码: {res.returncode}")
    except KeyboardInterrupt:
        sys.exit("\n⚠️ 本地监控已中断。云端任务可能仍在后台运行。")

    print("\n" + "=" * 70)
    print("🎉 云端任务已全部执行完成！" + (" (云端主机已触发关机)" if auto_shutdown else ""))


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
