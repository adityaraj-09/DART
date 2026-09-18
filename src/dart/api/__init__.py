"""HTTP / WebSocket gateway (CIP + OpenAI-compatible facade)."""

from dart.api.gateway import create_app, create_peer_app

__all__ = ["create_app", "create_peer_app"]
