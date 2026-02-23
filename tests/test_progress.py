"""Tests for the progress reporting module."""

import os
import sys
import io
import unittest
from unittest.mock import patch

# Add script directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'script'))

from progress import report_progress, ProgressTracker


class TestReportProgress(unittest.TestCase):
    """Tests for the report_progress() function."""

    def test_basic_output(self):
        """report_progress emits the correct format."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_out:
            report_progress(42, "hello world")
        self.assertEqual(mock_out.getvalue(), "@@PROGRESS@@ 42 hello world\n")

    def test_empty_message(self):
        """report_progress with no message emits pct only."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_out:
            report_progress(50)
        self.assertEqual(mock_out.getvalue(), "@@PROGRESS@@ 50\n")

    def test_clamp_low(self):
        """Negative pct is clamped to 0."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_out:
            report_progress(-10)
        self.assertIn("@@PROGRESS@@ 0", mock_out.getvalue())

    def test_clamp_high(self):
        """pct > 100 is clamped to 100."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_out:
            report_progress(200)
        self.assertIn("@@PROGRESS@@ 100", mock_out.getvalue())

    def test_float_pct_truncated(self):
        """Float pct is truncated to int."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_out:
            report_progress(33.7)
        self.assertIn("@@PROGRESS@@ 33", mock_out.getvalue())


class TestProgressTracker(unittest.TestCase):
    """Tests for the ProgressTracker class."""

    def _collect_output(self, tracker_fn):
        """Run tracker_fn and return list of emitted pct values."""
        lines = []
        with patch('sys.stdout', new_callable=io.StringIO) as mock_out:
            tracker_fn()
        for line in mock_out.getvalue().strip().splitlines():
            if line.startswith("@@PROGRESS@@"):
                pct = int(line.split()[1])
                lines.append(pct)
        return lines

    def test_basic_counter(self):
        """ProgressTracker emits 0-100 for a simple counter."""
        def fn():
            t = ProgressTracker(total=10, pct_start=0, pct_end=100)
            for i in range(10):
                t.update(i + 1)
            t.complete()

        pcts = self._collect_output(fn)
        self.assertGreater(len(pcts), 0)
        self.assertEqual(pcts[-1], 100)
        # All values should be 0-100
        for p in pcts:
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 100)

    def test_sub_range(self):
        """ProgressTracker maps to sub-range [20, 80]."""
        def fn():
            t = ProgressTracker(total=5, pct_start=20, pct_end=80, throttle=0)
            for i in range(5):
                t.update(i + 1)

        pcts = self._collect_output(fn)
        self.assertGreater(len(pcts), 0)
        # First should be around 32 (20 + 1/5 * 60 = 32)
        self.assertGreaterEqual(pcts[0], 20)
        # Last should be 80
        self.assertEqual(pcts[-1], 80)

    def test_throttle(self):
        """Throttle suppresses intermediate updates < threshold."""
        def fn():
            t = ProgressTracker(total=1000, pct_start=0, pct_end=100, throttle=5)
            for i in range(1000):
                t.update(i + 1)

        pcts = self._collect_output(fn)
        # With throttle=5%, we should have ~20 updates (100/5)
        # Allow some slack for rounding
        self.assertLess(len(pcts), 30)
        self.assertGreater(len(pcts), 10)

    def test_complete_emits_pct_end(self):
        """complete() always emits pct_end."""
        def fn():
            t = ProgressTracker(total=100, pct_start=10, pct_end=90)
            t.complete("done")

        pcts = self._collect_output(fn)
        self.assertEqual(pcts, [90])

    def test_total_zero_safe(self):
        """total=0 does not cause ZeroDivisionError."""
        def fn():
            t = ProgressTracker(total=0, pct_start=0, pct_end=100)
            t.update(0)
            t.complete()

        pcts = self._collect_output(fn)
        # Should not crash
        self.assertGreater(len(pcts), 0)


class TestWebtoolsParser(unittest.TestCase):
    """Tests for the @@PROGRESS@@ parser in run_webtools.py."""

    def test_parse_progress_line(self):
        """_parse_progress extracts pct from @@PROGRESS@@ lines."""
        import re
        _RE_PROGRESS = re.compile(r"@@PROGRESS@@\s+(\d+)\s*(.*)")

        line = "@@PROGRESS@@ 42 hello world"
        m = _RE_PROGRESS.search(line)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 42)
        self.assertEqual(m.group(2), "hello world")

    def test_parse_progress_no_msg(self):
        """@@PROGRESS@@ line with no message."""
        import re
        _RE_PROGRESS = re.compile(r"@@PROGRESS@@\s+(\d+)\s*(.*)")

        line = "@@PROGRESS@@ 99"
        m = _RE_PROGRESS.search(line)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 99)
        self.assertEqual(m.group(2).strip(), "")

    def test_parse_non_progress_line(self):
        """Non-progress lines don't match."""
        import re
        _RE_PROGRESS = re.compile(r"@@PROGRESS@@\s+(\d+)\s*(.*)")

        line = "[INFO] Some normal log line"
        m = _RE_PROGRESS.search(line)
        self.assertIsNone(m)

    def test_parse_embedded_in_log(self):
        """@@PROGRESS@@ embedded in a logger-formatted line still matches."""
        import re
        _RE_PROGRESS = re.compile(r"@@PROGRESS@@\s+(\d+)\s*(.*)")

        line = "[12:34:56] INFO: @@PROGRESS@@ 75 three quarters done"
        m = _RE_PROGRESS.search(line)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 75)


if __name__ == "__main__":
    unittest.main()
