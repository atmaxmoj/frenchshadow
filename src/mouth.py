"""Compatibility re-export for articulatory diagram generation.

New code should import from `src.diagrams` directly.
"""

from __future__ import annotations

from src.diagrams import diagram, has_diagram, phones_for_diagrams

__all__ = ["diagram", "has_diagram", "phones_for_diagrams"]
