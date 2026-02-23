"""Structured progress reporting for pipeline stages.

All progress is emitted as structured lines on stdout::

    @@PROGRESS@@ {pct} {msg}

- ``pct``: 0-100 integer (clamped)
- ``msg``: optional human-readable description
- Uses ``print(..., flush=True)`` to bypass logger formatting

The webtools dashboard has a single regex to parse these lines
and compute ETA from elapsed time — no per-stage parsing needed.
"""

from __future__ import annotations


def report_progress(pct: int, msg: str = "") -> None:
    """Emit a structured progress line to stdout."""
    pct = max(0, min(100, int(pct)))
    print(f"@@PROGRESS@@ {pct} {msg}".rstrip(), flush=True)


class ProgressTracker:
    """Counter-based progress tracker with sub-range mapping and throttling.

    Args:
        total: Expected number of items to process.
        pct_start: Output percentage at ``current=0``.
        pct_end: Output percentage at ``current=total``.
        throttle: Minimum percentage change before emitting (default 1%).

    Example::

        tracker = ProgressTracker(total=100, pct_start=10, pct_end=90)
        for i in range(100):
            do_work(i)
            tracker.update(i + 1, f"Processing item {i+1}")
        tracker.complete("Done")
    """

    def __init__(
        self,
        total: int = 100,
        pct_start: int = 0,
        pct_end: int = 100,
        throttle: int = 1,
    ) -> None:
        self.total = max(1, total)
        self.pct_start = pct_start
        self.pct_end = pct_end
        self.throttle = max(0, throttle)
        self._last_pct = -999  # force first emission

    def update(self, current: int, msg: str = "") -> None:
        """Report progress based on *current* out of *total*."""
        frac = max(0.0, min(1.0, current / self.total))
        pct = int(self.pct_start + frac * (self.pct_end - self.pct_start))
        pct = max(0, min(100, pct))
        if abs(pct - self._last_pct) >= self.throttle:
            self._last_pct = pct
            report_progress(pct, msg)

    def complete(self, msg: str = "") -> None:
        """Emit the final progress value (*pct_end*)."""
        pct = max(0, min(100, self.pct_end))
        self._last_pct = pct
        report_progress(pct, msg)
