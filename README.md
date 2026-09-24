# experiments

## 1. 环境配置

计算节点为 `air-node-03`，项目维护环境是 Python 3.10.20，现成环境位于
`/home/JJ_Group/lih2511/.conda/envs/verl`：

```bash
ssh air-node-03
cd /home/JJ_Group/lih2511/test/jev
conda activate /home/JJ_Group/lih2511/.conda/envs/verl
python --version
python check.py
```

需要重新创建环境时执行：

```bash
conda create -p /home/JJ_Group/lih2511/.conda/envs/verl python=3.10.20 -y
conda activate /home/JJ_Group/lih2511/.conda/envs/verl
python -m pip install -r requirements.txt
python -m pip install -e ./verl-agent --no-deps

export ALFWORLD_DATA=/home/JJ_Group/lih2511/.cache/alfworld
alfworld-download -f
python check.py
```

## 2. 预下载权重

正式实验只使用固定 revision 的 `Qwen/Qwen2.5-1.5B-Instruct`。launcher
启用了 Hugging Face 离线模式，因此必须在启动前完成下载：

```bash
conda activate /home/JJ_Group/lih2511/.conda/envs/verl

MODEL_SNAPSHOT=$(hf download Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --cache-dir /home/JJ_Group/lih2511/.cache/huggingface/hub \
  --quiet)

echo "$MODEL_SNAPSHOT"
test -f "$MODEL_SNAPSHOT/config.json"
```

`config/config.yaml` 中的 `model_path` 必须等于上述命令输出的 snapshot
绝对路径。当前共享目录已经包含该 revision。

## 3. 脚本启动

先在 `scripts/run_alfworld.sh` 顶部设置参与实验的方法，例如：

```bash
algos=(grpo jev gigpo graphgpo)
# 全部五种方法：algos=(grpo jev gigpo hgpo graphgpo)
```

正式运行前先做配置 dry-run：

```bash
cd /home/JJ_Group/lih2511/test/jev
CHECK_CONFIG_ONLY=true CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  scripts/run_alfworld.sh formal-v1
```

使用 8 张 H200 启动统一实验；包含 Jev 时需要提供 API key：

```bash
export TYPESAFE_API_KEY='<your-key>'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  scripts/run_alfworld.sh formal-v1
```

每个算法使用两张 GPU，最多四个算法并行；第五个算法进入下一批。脚本按
`config/config.yaml` 中的 seed 训练，保存所有 milestone checkpoint，并在训练
完成后自动测评 `valid_seen` 与 `valid_unseen`。结果写入
`runs/grpo-alfworld-formal-v1-<algo>-seed<seed>/`。

只启动单一方法时使用两张 GPU，并显式给出方法、run tag 和 seed：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  scripts/run_alfworld.sh grpo smoke-grpo-seed1 1
```
