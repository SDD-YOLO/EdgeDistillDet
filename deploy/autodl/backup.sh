#!/usr/bin/env bash
# 备份 EDGE_TEAM_DATA_ROOT 与 SQLite 文件（与 env 中 EDGE_DATABASE_URL 对应库文件）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-$SCRIPT_DIR/env.sh}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "缺少 $ENV_FILE"
  exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"
ROOT_DATA="${ROOT_DATA:-/root/autodl-tmp/edgedistilldet}"
mkdir -p "$ROOT_DATA/backups"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT_DATA/backups/edgedistilldet_${TS}.tar.gz"

ARGS=()
[[ -d "${EDGE_TEAM_DATA_ROOT:-}" ]] && ARGS+=("$EDGE_TEAM_DATA_ROOT")
# 默认与 env.example 一致：库文件在 $ROOT_DATA/saas.db
if [[ -f "$ROOT_DATA/saas.db" ]]; then
  ARGS+=("$ROOT_DATA/saas.db")
fi
if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "没有可备份的路径"
  exit 1
fi
tar -czvf "$OUT" "${ARGS[@]}"
echo "已生成: $OUT"
ls -lh "$OUT"
