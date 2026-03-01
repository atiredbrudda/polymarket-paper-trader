"""Test pm_sim.__main__ module."""

import subprocess
import sys


def test_main_module_runs():
    """python -m pm_sim executes without import errors."""
    result = subprocess.run(
        [sys.executable, "-m", "pm_sim", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "pm-sim" in result.stdout or "Usage" in result.stdout
