# Jev ALFWorld experiments

## 1. 环境配置

前置条件：Linux、可用的 NVIDIA 驱动、Conda 和 8 张 GPU。以下命令不依赖特定
用户名、Conda 目录或工作区路径：

```bash
git clone https://github.com/HuanLi0311/jev.git
cd jev

conda create -n rl python=3.10.20 -y
conda activate rl

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ./verl-agent --no-deps

alfworld-download -f
```

ALFWorld 默认下载到当前用户的 `~/.cache/alfworld`。如需共享缓存，可在下载和
运行前显式设置 `ALFWORLD_DATA`；launcher 会沿用该变量。

## 2. 预下载权重

训练使用固定 revision 的 `Qwen/Qwen2.5-1.5B-Instruct`。launcher 启用
Hugging Face 离线模式，因此需要提前下载：

```bash
conda activate rl
cd jev

MODEL_SNAPSHOT=$(hf download Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --quiet)

echo "$MODEL_SNAPSHOT"
test -f "$MODEL_SNAPSHOT/config.json"
python check.py
```

默认 Hugging Face 缓存下，`config/config.yaml` 已指向这个固定 revision。
如果设置了 `HF_HOME` 或使用 `--cache-dir`，将配置中的 `model_path` 改为
`MODEL_SNAPSHOT` 输出的绝对路径。

## 3. 脚本启动

当前维护的 suite 固定运行 Jev，单个 arm 使用全部 8 张 GPU。训练规模、seed、
checkpoint/evaluation milestone 和评测面板只由 `config/config.yaml` 定义；不要在
launcher 中复制这些值。当前 Jev 在整条轨迹结束后读取公开轨迹和 verified
outcome，并为同一批的所有轨迹并发请求评分。策略 credit 仍只施加到对应
transition 的 action token：`confidence * (2 * score - 1)`；CoT 默认关闭。

正式运行前先验证配置和调度，不会启动训练：

```bash
conda activate rl
cd jev

CHECK_CONFIG_ONLY=true CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  scripts/run_alfworld.sh formal-v1
```

使用 8 张 GPU 启动实验。先通过环境变量提供 API key；不要把 key 写入仓库、
README 或命令行参数：

```bash
export TYPESAFE_API_KEY='<your-key>'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  scripts/run_alfworld.sh formal-v1
```

脚本按 `config/config.yaml` 依次运行配置的 training seed，保存所有 milestone
checkpoint，并在每个 seed 训练结束后自动测评 `valid_seen` 和
`valid_unseen`。结果位于
`runs/grpo-alfworld-formal-v1-jev-seed<seed>/`。

如需只运行一个 seed，仍使用 8 张 GPU，并显式给出方法、run tag 和 seed：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  scripts/run_alfworld.sh jev smoke-jev-seed1 1
```

共享文件系统上如果 Ray worker 冷启动时注册超时，可使用 launcher 已有的校准开关：

```bash
PYTHON_NO_SITE=true STDLIB_SHM=true \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  scripts/run_alfworld.sh formal-v1
```
