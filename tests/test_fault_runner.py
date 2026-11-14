"""Offline fault checks must preserve their results and fail automation on errors."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from scripts import faults


class FaultRunnerTests(unittest.TestCase):
    def test_report_contains_fault_results_without_generating_audio(self):
        result = {"all_passed": True, "scenarios": {"fixture": {"passed": True}}}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(faults, "run_checks", return_value=result), mock.patch("builtins.print"):
            path = Path(directory, "report.json")
            self.assertEqual(faults.main(["--report", str(path)]), 0)
            self.assertEqual(json.loads(path.read_text())["lifecycle_fault_suite"], result)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_fault_returns_nonzero(self):
        with mock.patch.object(faults, "run_checks", return_value={"all_passed": False, "scenarios": {}}), \
             mock.patch("builtins.print"):
            self.assertEqual(faults.main([]), 1)
