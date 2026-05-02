from __future__ import annotations

from shapesplat.cleanup.generated_artifacts import (
    CleanupCandidate,
    delete_candidates_permanently,
    move_candidates_to_trash,
    save_cleanup_report,
    scan_generated_artifacts,
)
from shapesplat.cleanup.rules import load_cleanup_rules

__all__ = [
    "CleanupCandidate",
    "delete_candidates_permanently",
    "load_cleanup_rules",
    "move_candidates_to_trash",
    "save_cleanup_report",
    "scan_generated_artifacts",
]
