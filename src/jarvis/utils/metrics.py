import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class MetricsTracker:
    """Tracks latency metrics for JARVIS pipeline stages."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / "performance.jsonl"
        self._current_session_metrics: dict[str, Any] = {}
        self.start_time = time.time()
        self.total_tokens = 0

    def start_request(self, request_type: str = "text"):
        """Start a new request tracking session."""
        self._current_session_metrics = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": request_type,
            "stages": {},
            "total_latency_s": 0.0,
            "ttft_s": None,
            "ttfa_s": None,
            "status": "in_progress",
        }
        self._request_start_time = time.perf_counter()

    def record_stage(self, stage_name: str, duration_s: float):
        """Record the duration of a specific pipeline stage."""
        if not self._current_session_metrics:
            return
        self._current_session_metrics["stages"][stage_name] = duration_s

    def record_ttft(self, duration_s: float):
        """Record Time-To-First-Token."""
        if not self._current_session_metrics:
            return
        self._current_session_metrics["ttft_s"] = duration_s

    def record_ttfa(self, duration_s: float):
        """Record Time-To-First-Audio."""
        if not self._current_session_metrics:
            return
        self._current_session_metrics["ttfa_s"] = duration_s

    def end_request(self, status: str = "success"):
        """End the request tracking session and flush to disk."""
        if not self._current_session_metrics:
            return

        total_duration = time.perf_counter() - self._request_start_time
        self._current_session_metrics["total_latency_s"] = total_duration
        self._current_session_metrics["status"] = status

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._current_session_metrics) + "\n")
        except Exception as e:
            logger.error("Failed to write performance metrics", error=str(e))

        # Reset for next request
        self._current_session_metrics = {}

    def record_tokens(self, tokens: int):
        """Record token usage for global tracking."""
        self.total_tokens += tokens


# Global instance for easy import
metrics = MetricsTracker()
