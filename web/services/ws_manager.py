import asyncio
import logging
import time
from typing import Set, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class AsyncConnectionManager:
    """Async WebSocket manager using an internal asyncio.Queue for broadcast.

    Producers put messages to `queue` via `broadcast(message)` and a background
    task fans out messages to connected clients. This decouples producers from
    send latency and works well with FastAPI event loop.
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.message_cache: List[dict] = []
        self.cache_max_size = 200
        self.queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        if self._running:
            return
        self._running = True
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._broadcast_loop())
        logger.info("WS manager started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        logger.info("WS manager stopped")

    async def _broadcast_loop(self):
        while self._running:
            try:
                msg = await self.queue.get()
            except asyncio.CancelledError:
                break
            if msg is None:
                continue
            self._cache_message(msg)
            disconnected = []
            for conn in list(self.active_connections):
                try:
                    await conn.send_json(msg)
                except Exception as e:
                    logger.debug("WS send failed: %s", e)
                    disconnected.append(conn)
            for c in disconnected:
                await self.disconnect(c)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info("WS client connected (%d)", len(self.active_connections))
        welcome = {"type": "welcome", "timestamp": time.time(), "data": {"cached_messages": len(self.message_cache)}}
        await websocket.send_json(welcome)
        # replay recent
        for msg in self.message_cache[-20:]:
            try:
                await websocket.send_json(msg)
            except Exception:
                break

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info("WS client disconnected (%d)", len(self.active_connections))

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.debug("send personal failed: %s", e)
            await self.disconnect(websocket)

    async def broadcast(self, message: dict):
        await self.queue.put(message)

    def _cache_message(self, message: dict):
        self.message_cache.append({**message, "_ts": time.time()})
        if len(self.message_cache) > self.cache_max_size:
            self.message_cache = self.message_cache[-(self.cache_max_size // 2):]


# Singleton manager used by routes and training callbacks
manager = AsyncConnectionManager()
