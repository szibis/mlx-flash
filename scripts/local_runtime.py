#!/usr/bin/env python3
"""Manage the native macOS MLX server and its Docker Compose support stack."""

from __future__ import annotations

import json
import os
import platform
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / ".local" / "runtime"
PID_FILE = STATE_DIR / "server.json"
LOG_FILE = STATE_DIR / "server.log"
DEFAULT_MODEL = "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit"


def require_supported_host(system: str, machine: str) -> None:
    if system.lower() != "darwin":
        raise RuntimeError(
            "MLX with Metal needs native Apple Silicon macOS. Docker starts the support services; "
            "run this launcher directly on the Mac, not inside a Linux container."
        )
    if machine not in {"arm64", "aarch64"}:
        raise RuntimeError(
            "This Python is running as Intel/Rosetta. Open Terminal without Rosetta and try again."
        )


def server_command(python: str, model: str, host: str, port: int) -> list[str]:
    return [
        python, "-m", "mlx_flash_compress.serve", "--model", model,
        "--host", host, "--port", str(port), "--preload",
    ]


def managed_server_command(command: str, port: int) -> bool:
    try:
        args = shlex.split(command)
    except ValueError:
        return False
    return (
        "-m" in args
        and args[args.index("-m") + 1:args.index("-m") + 2] == ["mlx_flash_compress.serve"]
        and "--port" in args
        and args[args.index("--port") + 1:args.index("--port") + 2] == [str(port)]
    )


def _config() -> tuple[str, str, int, int]:
    model = os.environ.get("MLX_FLASH_MODEL", DEFAULT_MODEL).strip()
    host = os.environ.get("MLX_FLASH_HOST", "127.0.0.1").strip()
    try:
        port = int(os.environ.get("MLX_FLASH_PORT", "8080"))
        startup_timeout = int(os.environ.get("MLX_FLASH_STARTUP_TIMEOUT", "900"))
    except ValueError:
        raise RuntimeError("Port and startup timeout must be whole numbers.") from None
    if not model or len(model) > 300:
        raise RuntimeError("Choose a model name up to 300 characters.")
    if not 1 <= port <= 65535:
        raise RuntimeError("Port must be from 1 to 65535.")
    if not 30 <= startup_timeout <= 3600:
        raise RuntimeError("Startup timeout must be from 30 to 3600 seconds.")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("The one-click launcher only binds to this Mac for local privacy.")
    return model, host, port, startup_timeout


def _server_url(host: str, port: int, path: str = "/health") -> str:
    connect_host = "127.0.0.1" if host == "localhost" else host
    if ":" in connect_host and not connect_host.startswith("["):
        connect_host = f"[{connect_host}]"
    return f"http://{connect_host}:{port}{path}"


def _health(host: str, port: int) -> dict | None:
    try:
        with urllib.request.urlopen(_server_url(host, port), timeout=2) as response:
            value = json.loads(response.read(1024 * 1024))
        if isinstance(value, dict) and value.get("status") == "ok":
            return value
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None
    return None


def _port_open(host: str, port: int) -> bool:
    address = ("127.0.0.1" if host == "localhost" else host, port)
    family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.4)
        return connection.connect_ex(address) == 0


def _process_command(pid: int) -> str | None:
    result = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "command="],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _read_pid_record() -> dict | None:
    try:
        value = json.loads(PID_FILE.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("pid"), int):
            return value
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _managed_pid(record: dict, port: int) -> int | None:
    pid = record["pid"]
    command = _process_command(pid)
    if command and managed_server_command(command, port):
        return pid
    return None


def _docker_compose(*args: str, required: bool = True) -> bool:
    try:
        result = subprocess.run(
            ["docker", "compose", *args], cwd=ROOT, check=False,
            capture_output=not required, text=True,
        )
    except FileNotFoundError:
        if required:
            raise RuntimeError("Docker Desktop was not found. Install and open Docker Desktop, then try again.") from None
        return False
    if result.returncode and required:
        raise RuntimeError("Docker Compose could not start. Check Docker Desktop and try again.")
    return result.returncode == 0


def require_docker() -> None:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise RuntimeError("Docker Desktop was not found. Install and open Docker Desktop, then try again.") from None
    if result.returncode:
        raise RuntimeError("Docker Desktop is not ready. Open Docker Desktop, wait until it says Running, then try again.")


def _probe_metal() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core as mx; print(mx.default_device())"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    output = (probe.stdout + probe.stderr).strip()
    if probe.returncode:
        detail = output.splitlines()[-1] if output else "MLX could not initialize."
        raise RuntimeError(
            "MLX cannot access Metal on this Mac right now. Open this project in a normal Apple Silicon "
            f"Terminal and check Xcode/macOS setup. Details: {detail}"
        )


def _tail_log(lines: int = 35) -> str:
    try:
        return "\n".join(LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(No server log was written.)"


def up() -> int:
    require_supported_host(sys.platform, platform.machine())
    model, host, port, timeout = _config()
    if not _health(host, port) and _port_open(host, port):
        raise RuntimeError(f"Port {port} is already used by another program. Close it or choose MLX_FLASH_PORT.")
    if _health(host, port):
        print(f"MLX-Flash is already running: http://{host}:{port}/chat")
        subprocess.run(["open", f"http://{host}:{port}/chat"], check=False, capture_output=True)
        return 0

    _probe_metal()
    require_docker()
    print("Starting Docker support services…")
    _docker_compose("--profile", "monitoring", "up", "-d", "prometheus", "grafana")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    command = server_command(sys.executable, model, host, port)
    log_handle = LOG_FILE.open("ab", buffering=0)
    process = subprocess.Popen(
        command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log_handle,
        stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
    )
    log_handle.close()
    temporary = PID_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": process.pid, "model": model, "host": host, "port": port}) + "\n", encoding="utf-8")
    temporary.replace(PID_FILE)

    print(f"Loading {model}. This can take a few minutes the first time…")
    deadline = time.monotonic() + timeout
    last_message = time.monotonic()
    try:
        while time.monotonic() < deadline:
            health = _health(host, port)
            if health and health.get("model_loaded") is True:
                print(f"Ready: http://{host}:{port}/chat")
                print(f"Admin: http://{host}:{port}/admin · Logs: {LOG_FILE.relative_to(ROOT)}")
                print("Monitoring: http://localhost:3000 (Grafana) · http://localhost:9090 (Prometheus)")
                subprocess.run(["open", f"http://{host}:{port}/chat"], check=False, capture_output=True)
                return 0
            if _process_command(process.pid) is None:
                raise RuntimeError("The server stopped while loading the model. Recent log:\n" + _tail_log())
            if time.monotonic() - last_message >= 15:
                print("Still loading the model… please leave this window open.")
                last_message = time.monotonic()
            time.sleep(1)
        raise RuntimeError(f"The model did not become ready in {timeout} seconds. Recent log:\n" + _tail_log())
    except Exception:
        down_server()
        _docker_compose("--profile", "monitoring", "down", required=False)
        raise


def down_server() -> None:
    record = _read_pid_record()
    if not record:
        PID_FILE.unlink(missing_ok=True)
        return
    pid = _managed_pid(record, int(record.get("port", 8080)))
    PID_FILE.unlink(missing_ok=True)
    if pid is None:
        print("No matching MLX-Flash process is running.")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        if _process_command(pid) is None:
            return
        time.sleep(0.5)
    if managed_server_command(_process_command(pid) or "", int(record.get("port", 8080))):
        os.kill(pid, signal.SIGKILL)


def status() -> int:
    model, host, port, _ = _config()
    health = _health(host, port)
    if health and health.get("model_loaded") is True:
        print(f"READY — {health.get('model', model)} at http://{host}:{port}/chat")
        result = 0
    elif _port_open(host, port) and _managed_pid(_read_pid_record() or {"pid": -1}, port):
        print(f"STARTING — server answers on port {port}, model is not ready yet.")
        result = 0
    elif _port_open(host, port):
        print(f"PORT IN USE — another application is using {port}.")
        result = 1
    else:
        print("STOPPED — MLX-Flash is not running.")
        result = 0
    _docker_compose("--profile", "monitoring", "ps", required=False)
    return result


def down() -> int:
    down_server()
    _docker_compose("--profile", "monitoring", "down", required=False)
    print("MLX-Flash and its Docker support services are stopped.")
    return 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"check-host", "up", "status", "down"}:
        print("Usage: local_runtime.py check-host|up|status|down", file=sys.stderr)
        return 2
    action = sys.argv[1]
    try:
        if action == "check-host":
            require_supported_host(sys.platform, platform.machine())
            print("This Python is running natively on Apple Silicon macOS.")
            return 0
        if action == "up":
            return up()
        if action == "status":
            return status()
        return down()
    except RuntimeError as error:
        print(f"Could not start MLX-Flash: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
