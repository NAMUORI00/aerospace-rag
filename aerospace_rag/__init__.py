"""Aerospace local RAG package."""

from .private_overlay import PrivateOverlayStore
from .pipeline import ask, build_index

__all__ = ["ask", "build_index", "PrivateOverlayStore"]
