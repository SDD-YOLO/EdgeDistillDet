AutoDL 一键参考（SaaS + Redis 队列 + Worker）
============================================

1) 安装并启动 Redis：
   sudo apt update && sudo apt install -y redis-server
   若 redis-cli ping 报 Connection refused：容器内常无 systemd，勿用 systemctl。
   依次试：sudo service redis-server start
   或：redis-server --daemonize yes
   再 redis-cli ping（应返回 PONG）

2) 配置环境：
   cp deploy/autodl/env.example.sh deploy/autodl/env.sh
   编辑 env.sh：至少修改 EDGE_JWT_SECRET；确认 ROOT_DATA 指向数据盘（如 /root/autodl-tmp/edgedistilldet）

3) 开两个终端（或 tmux 两个 pane）：
   chmod +x deploy/autodl/start_api.sh deploy/autodl/start_worker.sh deploy/autodl/backup.sh
   ./deploy/autodl/start_api.sh
   ./deploy/autodl/start_worker.sh

4) AutoDL 控制台「自定义服务」映射端口与 EDGE_BACKEND_PORT 一致（默认 5000）

5) 验收：浏览器打开映射地址，GET /api/health ；注册账号 → 创建团队 → 选择团队后再用训练台

6) 备份：./deploy/autodl/backup.sh
   将生成的 .tar.gz 下载到本地保存

PostgreSQL 用户：自行安装并设置 EDGE_DATABASE_URL，备份改用 pg_dump，勿依赖本脚本中的 saas.db
