import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from lab_coordination import coordinate_lab


class LabCoordinationTests(unittest.TestCase):
    def test_disabled_does_not_call_helper(self):
        with coordinate_lab({}, run=lambda *a, **k: self.fail("unexpected helper")):
            pass

    def test_restores_after_success_or_failure_and_leaves_stopped_lab_off(self):
        for paused in (True, False):
            for failing in (True, False):
                with self.subTest(paused=paused, failing=failing), tempfile.TemporaryDirectory() as directory:
                    binary = Path(directory) / "sentinel-tools"
                    binary.write_text("fixture")
                    binary.chmod(0o700)
                    env = {"SENTINEL_CI_TOOLS_BIN": str(binary), "SENTINEL_CI_PROJECT_ROOT": directory, "SENTINEL_CI_LAB_ROOT": directory}
                    calls = []

                    def run(command, **kwargs):
                        calls.append(command)
                        return subprocess.CompletedProcess(command, 0, json.dumps({"paused": paused, "token": "owned-token" if paused else ""}))

                    try:
                        with coordinate_lab(env, run):
                            if failing:
                                raise RuntimeError("model failed")
                    except RuntimeError as error:
                        self.assertTrue(failing)
                        self.assertEqual(str(error), "model failed")
                    self.assertEqual(len(calls), 2 if paused else 1)
                    if paused:
                        self.assertIn("resume", calls[1])
                        self.assertEqual(calls[1][-1], "owned-token")

    def test_partial_configuration_and_symlink_binary_are_rejected(self):
        with self.assertRaises(RuntimeError):
            with coordinate_lab({"SENTINEL_CI_LAB_ROOT": "/private/tmp/fixture"}):
                self.fail("partial configuration accepted")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            target.write_text("fixture")
            target.chmod(0o700)
            binary = Path(directory) / "sentinel-tools"
            binary.symlink_to(target)
            env = {"SENTINEL_CI_TOOLS_BIN": str(binary), "SENTINEL_CI_PROJECT_ROOT": directory, "SENTINEL_CI_LAB_ROOT": directory}
            with self.assertRaises(RuntimeError):
                with coordinate_lab(env):
                    self.fail("symlink helper accepted")

    def test_restore_failure_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "sentinel-tools"
            binary.write_text("fixture")
            binary.chmod(0o700)
            env = {"SENTINEL_CI_TOOLS_BIN": str(binary), "SENTINEL_CI_PROJECT_ROOT": directory, "SENTINEL_CI_LAB_ROOT": directory}

            def run(command, **kwargs):
                if "resume" in command:
                    raise subprocess.CalledProcessError(1, command)
                return subprocess.CompletedProcess(command, 0, '{"paused":true,"token":"owned-token"}')

            with self.assertRaises(subprocess.CalledProcessError):
                with coordinate_lab(env, run):
                    pass


if __name__ == "__main__":
    unittest.main()
