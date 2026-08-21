"""针对「云端 GPU 实例按量计费」优化的三步式省钱训练工具。

工作流程：
  1. [无 GPU / 关机模式] python scripts/Tools/cloud_train.py push
     -> 先传代码和数据集到云端存储（不计 GPU 费用）。
  2. [网页开启 GPU 后]   python scripts/Tools/cloud_train.py run --stage full_sft --batch_size 224 --use_compile 1 [--shutdown]
     -> 在 GPU 开机期间仅跑训练并实时看日志。加 --shutdown 可在训练完后自动关机，防止持续扣费。
  3. [网页关闭 GPU 后]   python scripts/Tools/cloud_train.py pull
     -> 关机后，把训练好的模型权重从云端存储拉回本地（不计 GPU 费用）。

配置依赖 scripts/Tools/cloud_config.py（见 cloud_config.example.py）。
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
    raw_args = sys.argv[1:]
    if not raw_args or raw_args[0] in ('-h', '--help'):
        print(__doc__)
        print("可用命令：")
        print("  push                      传代码与数据到云端（开 GPU 前使用，省钱）")
        print("  run [训练参数] [--shutdown] 仅触发远程训练（开 GPU 后使用，支持训练完自动关机）")
        print("  pull                      拉回云端 models/ 权重（关 GPU 后使用，省钱）")
        return

    cfg = load_config()
    if not cfg['host']:
        sys.exit('错误：未配置云端主机。请在 scripts/Tools/cloud_config.py 设置 HOST（参考 cloud_config.example.py）')

    subcmd = raw_args[0]
    ssh_env = build_ssh(cfg['port'], cfg['password'])
    dst_prefix = f"{cfg['user']}@{cfg['host']}:{cfg['path']}"
    root = os.getcwd()

    # 1. 纯 CPU 模式上传
    if subcmd == 'push':
        print("=== [省钱模式 Step 1] 关机状态下传输数据/代码至云端 ===")
        for d in DEFAULT_DIRS:
            local_dir = os.path.join(root, d)
            remote_dir = f"{dst_prefix}/{d}"
            if os.path.exists(local_dir):
                run_rsync('push', local_dir, remote_dir, ssh_env, dry_run=False)
        print("✅ 数据与代码已成功同步至云端！现在可以在控制台开启 GPU 实例了。")

    # 2. 纯 CPU 模式拉回
    elif subcmd == 'pull':
        print("=== [省钱模式 Step 3] 关机状态下拉回模型权重 (models/) ===")
        local_models = os.path.join(root, 'models')
        remote_models = f"{dst_prefix}/models"
        run_rsync('pull', local_models, remote_models, ssh_env, dry_run=False)
        print("🎉 模型权重已拉回本地 models/ 目录！")

    # 3. GPU 开启阶段只跑训练
    elif subcmd == 'run':
        train_args = raw_args[1:]
        
        # 检查是否要求训练完自动关机（省钱利器）
        auto_shutdown = False
        if '--shutdown' in train_args:
            auto_shutdown = True
            train_args.remove('--shutdown')
        elif '--auto-shutdown' in train_args:
            auto_shutdown = True
            train_args.remove('--auto-shutdown')

        py_bin = 'python'
        if '--py_bin' in train_args:
            idx = train_args.index('--py_bin')
            py_bin = train_args[idx + 1]
            del train_args[idx:idx + 2]

        target_script = 'scripts/Trainer/train.py'
        if '--script' in train_args:
            idx = train_args.index('--script')
            target_script = train_args[idx + 1]
            del train_args[idx:idx + 2]

        print("=== [省钱模式 Step 2] GPU 已开启，开始云端训练 ===")
        train_args_str = shlex.join(train_args)
        
        # 组合远程训练命令，如果要求自动关机，在训练成功后执行 shutdown
        shutdown_cmd = " && (sudo shutdown || shutdown || poweroff)" if auto_shutdown else ""
        remote_cmd = f"cd {shlex.quote(cfg['path'])} && {py_bin} {target_script} {train_args_str}{shutdown_cmd}"
        full_cmd = build_ssh_cmd(cfg, remote_cmd)

        print(f"> 远程执行: {remote_cmd}\n")
        try:
            res = subprocess.run(full_cmd)
            if res.returncode != 0:
                sys.exit(f"\n错误：云端训练异常终止，退出码: {res.returncode}")
        except KeyboardInterrupt:
            sys.exit("\n本地已取消监控。云端训练仍在进行，请在平台控制台检查进度。")

        print("\n✅ 训练完成！" + (" 云端主机正准备自动关机..." if auto_shutdown else " 请在平台控制台关闭 GPU 实例。"))
        print("💡 关机后，在本地执行 `python scripts/Tools/cloud_train.py pull` 即可把权重拉回本地。")

    else:
        sys.exit(f"未知命令: '{subcmd}'。请使用 push / run / pull，或查看 --help。")


if __name__ == '__main__':
    main()
