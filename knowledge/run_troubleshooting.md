# eval_llm.py 运行问题记录

## 1. trainer_utils.py 语法错误

**报错**：
```
File "trainer/trainer_utils.py", line 215
    raw_model = model.module if isinstance(model, DistributedDataParallel) else 
                                                                                ^
SyntaxError: invalid syntax
```

**原因**：第215行 `else \` 后缺少表达式，反斜杠续行后跟的是注释，不是有效的 Python 表达式。

**修复**：将 `else \` 改为 `else model`。

**文件**：`trainer/trainer_utils.py:215`

---

## 2. PyTorch 2.6 torch.load weights_only 默认值变更

**报错**：
```
_pickle.UnpicklingError: Weights only load failed. In PyTorch 2.6, we changed
the default value of the `weights_only` argument in `torch.load` from `False` to `True`.
```

**原因**：PyTorch 2.6 将 `torch.load()` 的 `weights_only` 参数默认值从 `False` 改为 `True`，旧版 .pth 文件不兼容新默认值。

**修复**：在 `torch.load()` 调用中显式传入 `weights_only=False`。

```python
model.load_state_dict(torch.load(ckp, map_location=args.device, weights_only=False), strict=True)
```

**文件**：`eval_llm.py:39`

---

## 3. .pth 文件为 Git LFS 指针

**现象**：下载的 `*.pth` 文件只有几百字节，`file` 命令显示 `ASCII text`，内容是 Git LFS 指针。

**原因**：ModelScope/HuggingFace 使用 Git LFS 管理大文件，`git clone` 默认只下载指针文件，不下载实际权重。

**修复**：
```bash
cd MiniMind2-PyTorch
git lfs install
git lfs pull
```

**替代方案**（无需 git-lfs）：
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('gongjy/MiniMind2-PyTorch', cache_dir='./MiniMind2-PyTorch')"
```
