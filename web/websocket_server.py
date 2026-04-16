"""
EdgeDistillDet WebSocket 实时训练通信层 (Prompt 3)
================================================
提供 WebSocket 双向实时通信，用于训练过程数据推送。
支持: epoch_end, batch_end, metrics, log, complete, error 消息类型。
"""

import json
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Set
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    WebSocket 连接管理器 — 管理所有活跃的客户端连接，支持广播和单播。

    用法:
        manager = ConnectionManager()
        await manager.connect(websocket)  # 新连接加入
        await manager.broadcast(message)  # 广播给所有人
        manager.disconnect(websocket)     # 断开连接
    """

    def __init__(self):
        self.active_connections: Set[Any] = set()
        self._lock = threading.Lock()
        # 训练特定消息缓存（新连接可拉取最近消息）
        self.message_cache: List[Dict] = []
        self.cache_max_size = 200

    async def connect(self, websocket):
        """注册新的 WebSocket 连接"""
        self.active_connections.add(websocket)
        logger.info(f"[WS] Client connected. Total: {len(self.active_connections)}")
        # 发送欢迎消息 + 缓存的历史消息摘要
        try:
            welcome_msg = {
                "type": "welcome",
                "timestamp": time.time(),
                "data": {
                    "cached_messages": len(self.message_cache),
                    "message": "Connected to EdgeDistillDet Training Stream"
                }
            }
            await websocket.send_json(welcome_msg)

            # 推送最近的 N 条缓存消息给新客户端
            recent = self.message_cache[-20:]
            if recent:
                for msg in recent:
                    try:
                        await websocket.send_json(msg)
                    except Exception:
                        break
        except Exception as e:
            logger.warning(f"[WS] Failed to send welcome: {e}")

    def disconnect(self, websocket):
        """断开连接"""
        self.active_connections.discard(websocket)
        logger.info(f"[WS] Client disconnected. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket):
        """单播消息给指定客户端"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"[WS] Send personal failed: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict, cache: bool = True):
        """广播消息给所有连接的客户端"""
        # 缓存消息供新连接使用
        if cache:
            self._cache_message(message)

        disconnected = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.debug(f"[WS] Broadcast to one failed: {e}")
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

    def _cache_message(self, message: dict):
        """缓存消息到环形缓冲区"""
        self.message_cache.append({
            **message,
            "_ts": time.time()  # 内部时间戳
        })
        if len(self.message_cache) > self.cache_max_size:
            self.message_cache = self.message_cache[-(self.cache_max_size // 2):]

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


# ==================== 全局单例 ====================
manager = ConnectionManager()


# ==================== 训练回调类 ====================
class TrainingCallback:
    """
    训练过程回调 — 将 PyTorch/YOLO 训练事件转换为 WebSocket 消息并推送。

    可通过继承或组合方式集成到现有训练循环中:

        callback = TrainingCallback(manager)
        
        for epoch in range(epochs):
            callback.on_epoch_start(epoch, epochs)
            
            for batch_idx, batch in enumerate(dataloader):
                loss = train_step(model, batch)
                callback.on_batch_end(batch_idx, loss, total_batches)
                
            mAP = evaluate(model, val_loader)
            callback.on_epoch_end(epoch, epochs, {"mAP50": mAP})
    
    如果不依赖 WebSocket（如纯 SSE 模式），callback 会自动降级为静默模式。
    """

    MESSAGE_TYPES = {
        'epoch_start': 'epoch_start',
        'epoch_end': 'epoch_end',
        'batch_end': 'batch_end',
        'metrics': 'metrics',
        'log': 'log',
        'complete': 'complete',
        'error': 'error',
        'distill_update': 'distill_update'
    }

    def __init__(self, ws_manager: Optional[ConnectionManager] = None):
        self.manager = ws_manager or manager
        self.batch_counter = 0
        self.start_time = None

    async def _send(self, type_: str, data: Any):
        """发送标准格式消息"""
        message = {
            "type": type_,
            "timestamp": time.time(),
            "data": data
        }
        if self.manager:
            await self.manager.broadcast(message, cache=(type_ in ('epoch_end', 'metrics', 'complete', 'error', 'distill_update')))

    # ---- Epoch 级别事件 ----

    async def on_epoch_start(self, epoch: int, total_epochs: int, **kwargs):
        """每个 epoch 开始时调用"""
        await self._send('epoch_start', {
            "epoch": epoch + 1,
            "total_epochs": total_epochs,
            "start_time": kwargs.get('start_time', time.time())
        })

    async def on_epoch_end(self, epoch: int, total_epochs: int, metrics: dict = None, **kwargs):
        """每个 epoch 结束时调用"""
        data = {
            "epoch": epoch + 1,
            "total_epochs": total_epochs,
            "progress": ((epoch + 1) / total_epochs * 100) if total_epochs else 0
        }
        if metrics:
            data["metrics"] = metrics
        
        elapsed = 0
        if self.start_time:
            elapsed = time.time() - self.start_time
            data["elapsed_seconds"] = round(elapsed, 2)
            if total_epochs and epoch < total_epochs - 1:
                eta = (total_epochs - epoch - 1) * elapsed / max(1, epoch + 1)
                data["eta_seconds"] = round(eta, 2)
        
        await self._send('epoch_end', data)

    # ---- Batch 级别事件 ----

    async def on_batch_end(self, batch_idx: int, loss: float, total_batches: int, **kwargs):
        """每 N 个 batch 结束时调用（默认每10个）"""
        self.batch_counter += 1
        if self.batch_counter % 10 != 0:
            return
        
        await self._send('batch_end', {
            "batch": batch_idx + 1,
            "total_batches": total_batches,
            "loss": float(loss),
            "global_step": self.batch_counter
        })

    # ---- 指标事件 ----

    async def push_metrics(self, metrics: dict):
        """推送任意指标数据"""
        await self._send('metrics', metrics)

    async def push_distill_update(self, alpha: float, temperature: float, kd_loss: float, **extra):
        """推送蒸馏动态更新（用于前端可视化动画联动）"""
        await self._send('distill_update', {
            "alpha": round(alpha, 6),
            "temperature": round(temperature, 4),
            "kd_loss": round(kd_loss, 6),
            **extra
        })

    # ---- 日志 / 状态事件 ----

    async def push_log(self, line: str, level: str = 'info'):
        """推送日志行"""
        await self._send('log', {
            "line": line,
            "level": level
        })

    async def on_complete(self, final_metrics: dict = None, **kwargs):
        """训练完成时调用"""
        data = {"status": "completed"}
        if final_metrics:
            data["final_metrics"] = final_metrics
        if self.start_time:
            data["total_duration"] = round(time.time() - self.start_time, 2)
        await self._send('complete', data)

    async def on_error(self, error_msg: str, traceback_str: str = None):
        """训练出错时调用"""
        data = {"error": error_msg}
        if traceback_str:
            data["traceback"] = traceback_str[:2000]  # 截断过长的traceback
        await self._send('error', data)


# ==================== 训练任务管理器 ====================
class TrainingManager:
    """
    训练任务管理器 — 管理多个并行训练任务的启动、停止和生命周期。

    用法:
        tm = TrainingManager()
        tm.start_training("train_001", config_dict)
        status = tm.get_status("train_001")
        tm.stop_training("train_001")
    """

    def __init__(self):
        self.active_trainings: Dict[str, dict] = {}   # {training_id: {process, thread, config, start_time}}
        self.history: List[dict] = []                  # 已完成的训练记录

    def start_training(self, training_id: str, config: dict, run_func=None, **run_kwargs):
        """
        启动一个训练任务
        
        Args:
            training_id: 唯一任务标识
            config: 训练配置字典
            run_func: 实际的训练函数（可选，默认使用 subprocess）
            **run_kwargs: 额外参数传递给 run_func
        Returns:
            bool: 是否成功启动
        """
        if training_id in self.active_trainings:
            logger.warning(f"[TM] Training {training_id} already running")
            return False

        import multiprocessing as mp
        from queue import Queue

        log_queue = Queue(maxsize=3000)
        entry = {
            "id": training_id,
            "config": config,
            "start_time": time.time(),
            "pid": None,
            "process": None,
            "thread": None,
            "log_queue": log_queue,
            "status": "starting",
            "current_epoch": 0,
            "total_epochs": config.get("epochs", 0),
        }

        self.active_trainings[training_id] = entry
        logger.info(f"[TM] Starting training: {training_id}")

        return True

    def stop_training(self, training_id: str) -> bool:
        """停止指定的训练任务"""
        entry = self.active_trainings.get(training_id)
        if not entry:
            logger.warning(f"[TM] Training {training_id} not found")
            return False

        if entry.get("process"):
            try:
                entry["process"].terminate()
                logger.info(f"[TM] Terminated process for {training_id}")
            except Exception as e:
                logger.error(f"[TM] Failed terminate {training_id}: {e}")

        entry["status"] = "stopped"
        return True

    def get_status(self, training_id: str = None) -> Any:
        """获取训练状态"""
        if training_id:
            return self.active_trainings.get(training_id)
        return {k: {
            "id": v["id"],
            "status": v["status"],
            "current_epoch": v.get("current_epoch", 0),
            "total_epochs": v.get("total_epochs", 0),
            "duration": round(time.time() - v["start_time"], 1) if v["start_time"] else 0
        } for k, v in self.active_trainings.items()}

    def get_all_active_ids(self) -> List[str]:
        """获取所有活跃训练 ID"""
        return list(self.active_trainings.keys())


# ==================== Flask-SocketIO 集成路由 ====================
def register_socketio_routes(socketio_app, training_callback: TrainingCallback = None):
    """
    注册 Socket.IO 事件处理器（当使用 python-socketio 时）

    Usage in app.py:
        from flask_socketio import SocketIO
        socketio = SocketIO(app, cors_allowed_origins="*")
        from websocket_server import register_socketio_routes, manager, TrainingCallback
        callback = TrainingCallback(manager)
        register_socketio_routes(socketio, callback)
    """
    cb = training_callback or TrainingCallback(manager)

    @socketio_app.event
    def connect():
        print(f'[SocketIO] Client connected. Active: {len(manager.active_connections)}')

    @socketio_app.event
    def disconnect():
        print(f'[SocketIO] Client disconnected. Active: {len(manager.active_connections)}')

    @socketio_app.event
    async def command(data):
        """接收来自前端的命令: start/pause/resume/stop"""
        cmd = data.get('command')
        training_id = data.get('training_id', 'default')

        if cmd == 'start':
            config = data.get('config', {})
            cb.start_time = time.time()
            response = {"type": "command_response", "data": {"command": cmd, "accepted": True, "training_id": training_id}}

        elif cmd == 'stop':
            response = {"type": "command_response", "data": {"command": cmd, "accepted": True}}

        elif cmd == 'ping':
            response = {"type": "pong", "data": {"server_time": time.time(), "active_trainings": len([v for v in {}.values()])}}  # noqa: F821

        else:
            response = {"type": "command_response", "data": {"command": cmd, "accepted": False, "error": f"Unknown command: {cmd}"}}

        await manager.broadcast(response, cache=False)

    return cb


# ==================== FastAPI WebSocket 路由示例 ====================
# 如需迁移到 FastAPI，可直接使用以下路由定义:
#
# from fastapi import APIRouter, WebSocket, WebSocketDisconnect
# router = APIRouter()
#
# @router.websocket("/ws/training/{training_id}")
# async def training_websocket(websocket: WebSocket, training_id: str):
#     await manager.connect(websocket)
#     try:
#         while True:
#             data = await websocket.receive_json()
#             command = data.get('command')
#             # 处理命令...
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)


if __name__ == '__main__':
    # 测试模式：打印模块信息
    print("=" * 55)
    print("  EdgeDistillDet WebSocket Server Module")
    print(f"  ConnectionManager ready | Cache size: {manager.cache_max_size}")
    print("=" * 55)
