#!/bin/bash

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# This script was last updated/tuned on Jan 17, 2026.

# Run as:
# bash runs/runcpu.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# You may also want to run this script manually and one by one, copy pasting commands into your terminal.

# Where to pull datasets/tokenizers from. Set to a mirror to avoid huggingface.co, e.g.
# HF_ENDPOINT=https://hf-mirror.com bash runs/runcpu.sh
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
echo "Using hub endpoint: $HF_ENDPOINT"

# Proxy for reaching the endpoint above.
# Override with e.g. NANOCHAT_PROXY=http://127.0.0.1:7890 bash runs/runcpu.sh
# Set NANOCHAT_PROXY="" to disable and go direct.
NANOCHAT_PROXY="${NANOCHAT_PROXY-${https_proxy:-http://127.0.0.1:1087}}"
if [ -n "$NANOCHAT_PROXY" ]; then
    echo "Using proxy: $NANOCHAT_PROXY"
    export http_proxy="$NANOCHAT_PROXY"
    export https_proxy="$NANOCHAT_PROXY"
    export HTTP_PROXY="$NANOCHAT_PROXY"
    export HTTPS_PROXY="$NANOCHAT_PROXY"
    export no_proxy="localhost,127.0.0.1"
    # a mirror is typically local, so routing it through the proxy only adds latency
    case "$HF_ENDPOINT" in
        https://huggingface.co) ;;
        *)  hf_host="${HF_ENDPOINT#*://}"   # strip scheme
            hf_host="${hf_host%%/*}"        # strip path
            no_proxy="$no_proxy,$hf_host" ;;
    esac
    export NO_PROXY="$no_proxy"
    export no_proxy
    # urllib honors the lowercase vars above but would also pick up a socks5
    # ALL_PROXY, which it cannot speak without PySocks. Drop it.
    unset ALL_PROXY all_proxy
fi

# all the setup stuff
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# train tokenizer on ~2B characters (~34 seconds on my MacBook Pro M3 Max)
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# train a small 6 layer model
# I tuned this run to complete in about 30 minutes on my MacBook Pro M3 Max.
# To get better results, try increasing num_iterations, or get other ideas from your favorite LLM.
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --run=$WANDB_RUN
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16

# SFT (~10 minutes on my MacBook Pro M3 Max)
python -m scripts.chat_sft \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=$WANDB_RUN

# Chat with the model over CLI
# The model should be able to say that it is Paris.
# It might even know that the color of the sky is blue.
# Sometimes the model likes it if you first say Hi before you ask it questions.
# python -m scripts.chat_cli -p "What is the capital of France?"
