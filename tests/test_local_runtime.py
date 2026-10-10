"""Tests for the one-command native MLX + Docker runtime launcher."""

import unittest
from unittest.mock import patch

from scripts.local_runtime import _config, managed_server_command, require_docker, require_supported_host, server_command


class HostValidationTests(unittest.TestCase):
    def test_accepts_native_apple_silicon_macos(self):
        require_supported_host("Darwin", "arm64")

    def test_rejects_linux_container_with_actionable_message(self):
        with self.assertRaisesRegex(RuntimeError, "native Apple Silicon macOS"):
            require_supported_host("Linux", "aarch64")

    def test_rejects_rosetta_python(self):
        with self.assertRaisesRegex(RuntimeError, "Rosetta"):
            require_supported_host("Darwin", "x86_64")


class NativeServerCommandTests(unittest.TestCase):
    def test_builds_native_server_command_with_preload(self):
        self.assertEqual(
            server_command(".venv/bin/python", "mlx-community/tiny-model", "127.0.0.1", 8080),
            [
                ".venv/bin/python", "-m", "mlx_flash_compress.serve",
                "--model", "mlx-community/tiny-model", "--host", "127.0.0.1",
                "--port", "8080", "--preload",
            ],
        )

    def test_matches_only_expected_server_and_port(self):
        command = "/repo/.venv/bin/python -m mlx_flash_compress.serve --host 127.0.0.1 --port 8080 --preload"
        self.assertTrue(managed_server_command(command, 8080))
        self.assertFalse(managed_server_command(command, 8081))
        self.assertFalse(managed_server_command("python -m http.server 8080", 8080))

    def test_server_configuration_is_local_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            model, host, port, timeout = _config()
        self.assertEqual(model, "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit")
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 8080)
        self.assertGreaterEqual(timeout, 30)

    def test_server_rejects_non_local_bind_address(self):
        with patch.dict("os.environ", {"MLX_FLASH_HOST": "0.0.0.0"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "only binds to this Mac"):
                _config()


class DockerReadinessTests(unittest.TestCase):
    @patch("scripts.local_runtime.subprocess.run")
    def test_explains_how_to_recover_when_docker_desktop_is_unavailable(self, run):
        run.return_value.returncode = 1
        with self.assertRaisesRegex(RuntimeError, "Open Docker Desktop"):
            require_docker()


if __name__ == "__main__":
    unittest.main()
