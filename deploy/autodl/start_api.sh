#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${1:-$SCRIPT_DIR/env.sh}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "缺少 $ENV_FILE，请: cp deploy/autodl/env.example.sh deploy/autodl/env.sh 并编辑"
  exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"
cd "$REPO_ROOT"
exec python -u web/app.py
