#!/usr/bin/env bash
set -euo pipefail

target=${1:-/dev/shm/verl-agent-import-overlay}
python_bin=${PYTHON_BIN:-python}
site=$("$python_bin" -c 'import site; print(site.getsitepackages()[0])')
repo_root=$(cd "$(dirname "$0")/../.." && pwd)

mkdir -p "$target"

# ponytail: this is the smallest package set observed on the NFS import path.
# Add a package only when a concurrent-import failure identifies it.
packages=(
    torch functorch torchgen triton tensordict transformers tokenizers numpy
    fsspec huggingface_hub peft accelerate safetensors
)

for package in "${packages[@]}"; do
    cp -a "$site/$package" "$target/"
    for metadata in "$site"/"$package"-*.dist-info; do
        test ! -e "$metadata" || cp -a "$metadata" "$target/"
    done
done

if [[ ! -e "$target/nvidia" ]]; then
    ln -s "$site/nvidia" "$target/nvidia"
fi

PYTHONPATH="$target:$repo_root${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -c \
    'from verl.workers.fsdp_workers import ActorRolloutRefWorker; import fsspec, huggingface_hub, peft; print("overlay import check: ok")'

printf '%s\n' "$target"
