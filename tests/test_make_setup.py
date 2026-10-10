"""Exercise setup recovery without downloading packages or touching a real venv."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("make"), "make is required")
class SetupRecoveryTests(unittest.TestCase):
    def run_setup(self, version, has_pip):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts/local_runtime.py").write_text("# host check stub\n")
            (root / ".venv/bin").mkdir(parents=True)
            (root / ".venv/version").write_text(version)
            (root / ".venv/.mlxflash-ready").touch()
            if has_pip:
                (root / ".venv/pip-ready").touch()
            wrapper = (
                f"#!{sys.executable}\n"
                + """
import pathlib, sys
root = pathlib.Path.cwd()
args = sys.argv[1:]
is_base = pathlib.Path(sys.argv[0]).name == "base-python"
with (root / "calls").open("a") as f: f.write(str((is_base, args)) + "\\n")
if args[0] == "-c":
    print("3.13" if is_base else (root / ".venv/version").read_text())
elif args[:2] == ["-m", "venv"]:
    (root / ".venv/bin").mkdir(parents=True)
    target = root / ".venv/bin/python"
    target.write_text(pathlib.Path(sys.argv[0]).read_text()); target.chmod(0o755)
    (root / ".venv/version").write_text("3.13")
elif args[:3] == ["-m", "pip", "--version"]:
    sys.exit(0 if (root / ".venv/pip-ready").exists() else 1)
elif args[:2] == ["-m", "ensurepip"]:
    (root / ".venv/pip-ready").touch()
elif args[:3] == ["-m", "pip", "install"]:
    pass
else: sys.exit(2)
"""
            )
            for path in (root / "base-python", root / ".venv/bin/python"):
                path.write_text(wrapper)
                path.chmod(0o755)
            env = dict(os.environ)
            env.pop("MAKEFLAGS", None)
            result = subprocess.run(
                ["make", "-f", str(ROOT / "Makefile"), "setup", f"BASE_PYTHON={root / 'base-python'}"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / ".venv/.mlxflash-ready").exists())
            return (root / "calls").read_text()

    def test_recreates_wrong_python_and_restores_missing_pip(self):
        calls = self.run_setup("3.14", False)
        self.assertIn("'venv'", calls)
        self.assertIn("'ensurepip'", calls)
        self.assertIn("'install'", calls)

    def test_restores_pip_without_rebuilding_matching_environment(self):
        calls = self.run_setup("3.13", False)
        self.assertNotIn("'venv'", calls)
        self.assertIn("'ensurepip'", calls)

    def test_reuses_healthy_environment_without_reinstalling(self):
        calls = self.run_setup("3.13", True)
        self.assertNotIn("'venv'", calls)
        self.assertNotIn("'ensurepip'", calls)
        self.assertNotIn("'install'", calls)
