# 复制为 env.sh 后修改：cp deploy/autodl/env.example.sh deploy/autodl/env.sh
# 使用：source deploy/autodl/env.sh

# --- SaaS 与密钥 ---
export EDGE_SAAS_ENABLED=1
export EDGE_USE_TRAINING_QUEUE=1
# 务必改为长随机串；勿提交 env.sh 到 git
export EDGE_JWT_SECRET='CHANGE_ME_TO_LONG_RANDOM_STRING'

# --- Redis（本机默认）---
export EDGE_REDIS_URL='redis://127.0.0.1:6379/0'

# --- 数据目录（优先 AutoDL 数据盘）---
export ROOT_DATA="${ROOT_DATA:-/root/autodl-tmp/edgedistilldet}"
mkdir -p "$ROOT_DATA/team_data" "$ROOT_DATA/backups"
export EDGE_TEAM_DATA_ROOT="$ROOT_DATA/team_data"

# --- 数据库：SQLite（先跑通）---
export EDGE_DATABASE_URL="sqlite:///${ROOT_DATA}/saas.db"

# --- 若改用 PostgreSQL，注释上一行并取消下面注释 ---
# export EDGE_DATABASE_URL='postgresql+psycopg2://用户:密码@127.0.0.1:5432/库名'

# --- 监听（AutoDL 自定义服务映射端口）---
export EDGE_BACKEND_HOST=0.0.0.0
export EDGE_BACKEND_PORT=5000

# --- 可选：对外 URL（OAuth 回调等）---
# export EDGE_PUBLIC_APP_URL='https://你的映射地址'

# 项目根：在 env.sh 末尾可设置（或启动脚本里 cd）
# export EDGE_DISTILL_ROOT=/path/to/EdgeDistillDet
