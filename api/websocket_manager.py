"""WebSocket manager for real-time artifact updates.

Broadcasts artifact updates to all connected clients for LiveDesign-style
real-time collaboration.
"""

from typing import Dict, Set
from fastapi import WebSocket
import json
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket, user_id: str = None, workspace_id: str = None):
        """Register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "workspace_id": workspace_id,
        }
        logger.info(f"WebSocket connected: {user_id} in workspace {workspace_id}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        self.connection_metadata.pop(websocket, None)
        logger.info("WebSocket disconnected")

    async def broadcast(self, message: Dict, workspace_id: str = None):
        """Broadcast message to all connections (or filtered by workspace).

        Args:
            message: JSON-serializable dict
            workspace_id: If provided, only send to connections in this workspace
        """
        if not self.active_connections:
            return  # No clients connected

        disconnected = set()

        for connection in self.active_connections:
            # Filter by workspace if specified
            if workspace_id:
                meta = self.connection_metadata.get(connection, {})
                if meta.get("workspace_id") != workspace_id:
                    continue

            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.add(connection)

        # Clean up dead connections
        for conn in disconnected:
            self.disconnect(conn)

    async def send_to_user(self, user_id: str, message: Dict):
        """Send message to specific user's connections."""
        for connection, meta in self.connection_metadata.items():
            if meta.get("user_id") == user_id:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to user {user_id}: {e}")


# Global singleton instance
ws_manager = WebSocketManager()
