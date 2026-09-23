import subprocess, sys


def test_cli_help_lists_commands():
    out = subprocess.run([sys.executable, "-m", "comfy_agent", "--help"], capture_output=True, text=True)
    assert out.returncode == 0
    for c in ("catalog", "run", "baseline", "grid", "eval"):
        assert c in out.stdout
