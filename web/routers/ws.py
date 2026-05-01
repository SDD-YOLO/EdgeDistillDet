from __future__ import annotations

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from web.services.ws_manager import manager

router = APIRouter()


@router.websocket("/ws/train/{training_id}")
async def training_websocket(websocket: WebSocket, training_id: str):
    await manager.connect(websocket)
    try:
        while True:
            try:
                text = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            # Simple command handling: accept JSON commands and ack
            try:
                payload = json.loads(text)
                cmd = payload.get("command")
                if cmd == "ping":
                    await manager.send_personal_message({"type": "pong"}, websocket)
                else:
                    await manager.send_personal_message({"type": "ack", "command": cmd}, websocket)
            except Exception:
                await manager.send_personal_message({"type": "ack", "raw": text}, websocket)
    finally:
        await manager.disconnect(websocket)
