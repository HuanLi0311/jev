# Jev ALFWorld experiments

## 1. 环境配置

前置条件：Linux、可用的 NVIDIA 驱动、Conda 和 8 张 GPU。以下命令不依赖特定
用户名、Conda 目录或工作区路径：

```bash
git clone https://github.com/HuanLi0311/jev.git
cd jev

conda create -n jev-alfworld python=3.10.20 -y
conda activate jev-alfworld

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
conda activate jev-alfworld
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

在 `scripts/run_alfworld.sh` 顶部选择参与实验的方法：

```bash
algos=(grpo jev gigpo graphgpo)
# 全部五种方法：algos=(grpo jev gigpo hgpo graphgpo)
```

正式运行前先验证配置和调度，不会启动训练：

```bash
conda activate jev-alfworld
cd jev

CHECK_CONFIG_ONLY=true CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  scripts/run_alfworld.sh formal-v1
```

使用 8 张 GPU 启动统一实验。包含 Jev 时先通过环境变量提供 API key：

```bash
export TYPESAFE_API_KEY='<your-key>'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  scripts/run_alfworld.sh formal-v1
```

每个算法使用两张 GPU，最多四个算法并行；第五个算法进入下一批。脚本按
`config/config.yaml` 训练，保存所有 milestone checkpoint，并在训练结束后
自动测评 `valid_seen` 和 `valid_unseen`。结果位于
`runs/grpo-alfworld-formal-v1-<algo>-seed<seed>/`。

单独启动一种方法时使用两张 GPU，并显式给出方法、run tag 和 seed：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  scripts/run_alfworld.sh grpo smoke-grpo-seed1 1
```
