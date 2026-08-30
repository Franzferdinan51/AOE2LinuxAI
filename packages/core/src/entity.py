"""DetectedEntity dataclass — the schema YOLO inference emits and synthetic
perception projects into.

Lives in core because both `detection` (real inference) and `evaluation`
(synth render) produce it, and `gameplay_agent` consumes it. Putting it
here breaks the would-be cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectedEntity:
    """Represents a detected game entity."""

    id: str
    class_name: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    confidence: float
    area: float = field(default=0)

    def to_dict(self) -> dict:
        """Convert to dictionary for LLM context."""
        return {
            "id": self.id,
            "class": self.class_name,
            "bbox": list(self.bbox),
            "center": self.center,
            "confidence": self.confidence,
        }
