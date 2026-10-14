#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export MODELSCOPE_CACHE="$ROOT/.modelscope_cache"
export NASAL_MVP_DEVICE="${NASAL_MVP_DEVICE:-cpu}"
echo "启动本机界面 http://127.0.0.1:7860  （不上传公网）"
exec "$ROOT/.venv/bin/python" -m nasal_mvp ui --port 7860
