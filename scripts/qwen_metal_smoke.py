#!/usr/bin/env python3
"""Real, offline Qwen generation on a trusted Apple Silicon CI host.

Also used by Sentinel: pass --gateway with its freshly built gateway binary.
Only owned process groups are stopped; the interactive lab is never touched.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import socket
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

MARKER = "QWEN_ROLE_READY"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@contextmanager
def machine_lock(path, timeout=1800):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError("CI lock must be a regular file")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Timed out waiting for the shared Qwen CI lock") from None
                time.sleep(0.1)
        yield
    finally:
        os.close(fd)  # Do not unlink: waiting clients must share the same inode.


def request(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with OPENER.open(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # Backend errors can contain machine paths or prompts; keep raw bodies private.
        raise RuntimeError(f"HTTP generation/readiness request failed ({error.code})") from None


def wait_ready(process, url, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Owned server exited before becoming ready")
        try:
            health = request(url, timeout=min(2, max(0.01, deadline - time.monotonic())))
            if health.get("status") == "ok":
                return health
        except (OSError, ValueError, RuntimeError):
            pass
        time.sleep(0.1)
    raise RuntimeError("Timed out waiting for owned server readiness")


@contextmanager
def owned_server(command, log, env):
    with open(log, "w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=stream,
                                   stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        try:
            yield process
        finally:
            # Kill the group even if its leader exited and left descendants.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                if sig == signal.SIGTERM:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            process.wait(timeout=5)


def validate_model(value):
    model = Path(value)
    if not model.is_absolute():
        raise RuntimeError("Model paths must be absolute cached directories")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        file = model / name
        if not file.is_file() or not file.stat().st_size:
            raise RuntimeError(f"Cached model is missing nonempty {name}")
    weights = list(model.glob("*.safetensors"))
    if not weights or any(not file.stat().st_size for file in weights):
        raise RuntimeError("Cached model has missing/empty weights")
    index = model / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        if not shards or any(Path(s).name != s or not (model / s).is_file()
                             or not (model / s).stat().st_size for s in shards):
            raise RuntimeError("Cached model index has missing/invalid shards")
    elif any("-of-" in file.name for file in weights):
        raise RuntimeError("Sharded model requires a weights index")
    template_file = model / "chat_template.jinja"
    template = (template_file.read_text() if template_file.exists() else
                json.loads((model / "tokenizer_config.json").read_text()).get("chat_template"))
    if not isinstance(template, str) or "enable_thinking" not in template:
        raise RuntimeError("Cached model must have a Qwen thinking template")
    return model


def preflight(executable, models):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Real Qwen CI requires macOS Apple Silicon")
    python = executable.parent / "python"
    if not executable.is_file() or not os.access(executable, os.X_OK) or not python.is_file():
        raise RuntimeError("QWEN_MLX_FLASH_BIN must name an installed venv executable")
    check = subprocess.run([str(python), "-c", "import mlx.core as mx; assert mx.metal.is_available(), 'Metal unavailable'"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check.returncode:
        raise RuntimeError("Native runtime cannot initialize MLX Metal; real smoke failed preflight")
    for model in models:
        validate_model(str(model))
    preflight_ports((19190, 19191))


def preflight_ports(ports):
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sockets.append(sock)
            # HTTPServer uses REUSEADDR: sequential phases must tolerate its
            # recently closed connections in TIME_WAIT. Never use REUSEPORT;
            # a live listener still makes this exclusive bind fail.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
    except OSError:
        raise RuntimeError("CI ports are occupied; existing services were not stopped") from None
    finally:
        for sock in sockets:
            sock.close()


def final_text(text, thinking):
    if thinking:
        if "</think>" not in text:
            raise RuntimeError("Qwen thinking generation did not finish reasoning")
        text = text.split("</think>", 1)[1]
    elif "</think>" in text:
        text = text.split("</think>", 1)[1]
    if text.strip() != MARKER:
        raise RuntimeError("Qwen generation failed the exact final-answer marker")


def generate_chat(thinking):
    result = request("http://127.0.0.1:19191/v1/chat/completions", {
        "model": "local", "messages": [{"role": "user", "content": f"Reply with exactly {MARKER} and no other final text."}],
        "max_tokens": 8192 if thinking else 256, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": thinking}, "stream": False})
    choice = result["choices"][0]
    budget = 8192 if thinking else 256
    if choice.get("finish_reason") != "stop" or result.get("usage", {}).get("completion_tokens", budget) >= budget:
        raise RuntimeError("Qwen chat generation was truncated or lacked token accounting")
    final_text(choice["message"]["content"], thinking)
    return result.get("usage", {})


def generate_role(role, tool=False):
    payload = {"model": "sentinel-" + role, "max_tokens": 8192 if role == "opus" else 512,
               "messages": [{"role": "user", "content": f"Reply with exactly {MARKER} and no other final text."}]}
    if tool:
        payload["messages"][0]["content"] = f"Call record_marker once with marker {MARKER}. Do not answer in text."
        payload.update(tools=[{"name": "record_marker", "description": "Record a smoke-test marker; no side effects.",
                              "input_schema": {"type": "object", "properties": {"marker": {"type": "string", "enum": [MARKER]}},
                                               "required": ["marker"], "additionalProperties": False}}],
                       tool_choice={"type": "tool", "name": "record_marker"})
    result = request("http://127.0.0.1:19190/v1/messages", payload)
    if tool:
        calls = [b for b in result.get("content", []) if b.get("type") == "tool_use"]
        if (result.get("stop_reason") != "tool_use" or len(calls) != 1 or
                calls[0].get("name") != "record_marker" or calls[0].get("input") != {"marker": MARKER}):
            raise RuntimeError("Sentinel tool bridge failed the validated marker call")
    else:
        if result.get("stop_reason") != "end_turn":
            raise RuntimeError("Sentinel role generation did not finish")
        final_text("".join(b.get("text", "") for b in result.get("content", []) if b.get("type") == "text"), False)
    return result.get("usage", {})


def run(args, results, raw):
    executable = Path(os.environ["QWEN_MLX_FLASH_BIN"])
    models = [("small", Path(os.environ["QWEN_SMALL_MODEL_PATH"]))]
    if args.gateway or os.environ.get("QWEN_RUN_LARGE", "true").lower() == "true":
        models.append(("large", Path(os.environ["QWEN_LARGE_MODEL_PATH"])))
    preflight(executable, [model for _, model in models])
    env = {key: os.environ[key] for key in ("HOME", "PATH", "LANG", "LC_ALL", "TMPDIR") if key in os.environ}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    for name, model in models:
        print(f"Starting owned {name} Qwen runtime on isolated CI ports.", flush=True)
        command = [str(executable), "--model", str(model), "--host", "127.0.0.1", "--port", "19191",
                   "--speculative", "none", "--request-timeout", "600"]
        with owned_server(command, raw / f"{name}-runtime.log", env) as process:
            health = wait_ready(process, "http://127.0.0.1:19191/health")
            if "enable_thinking" not in health.get("capabilities", {}).get("chat_template_kwargs", []):
                raise RuntimeError("Native runtime does not advertise Qwen thinking profiles")
            results["health"].append({"model_size": name, "status": health["status"], "capabilities": health["capabilities"]})
            if args.gateway:
                upstream = "http://127.0.0.1:19191/v1"
                # Only the model under test is loaded. Each tested role still goes
                # through the real RoleRouter (including Opus's thinking profile).
                gateway = [args.gateway, "--listen", "127.0.0.1:19190", "--upstream", upstream,
                           "--role-haiku-upstream", upstream, "--role-sonnet-upstream", upstream,
                           "--role-opus-upstream", upstream, "--claude-max-tokens", "8192", "--timeout", "10m"]
                with owned_server(gateway, raw / f"{name}-gateway.log", env) as gateway_process:
                    state = wait_ready(gateway_process, "http://127.0.0.1:19190/health")
                    if not state.get("capabilities", {}).get("claude_roles"):
                        raise RuntimeError("Gateway did not advertise all Claude roles")
                    results["health"].append({"model_size": name, "scope": "gateway", "capabilities": state["capabilities"]})
                    for role in (["haiku"] if name == "small" else ["sonnet", "opus"]):
                        print(f"Generating real Messages response: {role}.", flush=True)
                        results["checks"].append({"role": role, "model_size": name, "usage": generate_role(role), "passed": True})
                    if name == "large":
                        print("Generating validated Sonnet tool marker (no tool execution).", flush=True)
                        results["checks"].append({"role": "sonnet", "tool_bridge": True, "usage": generate_role("sonnet", tool=True), "passed": True})
            else:
                for thinking in (False, True):
                    print(f"Generating real Chat Completions response: {name}, thinking={thinking}.", flush=True)
                    results["checks"].append({"model_size": name, "thinking": thinking, "usage": generate_chat(thinking), "passed": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", help="Fresh Sentinel gateway binary (otherwise test native Chat Completions)")
    parser.add_argument("--artifacts", default="qwen-metal-artifacts")
    args = parser.parse_args()
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    results = {"passed": False, "health": [], "checks": []}
    def interrupted(signum, frame):
        raise RuntimeError("Qwen smoke interrupted; cleaning up owned processes")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        with machine_lock(os.environ.get("QWEN_CI_LOCK_PATH", "/private/tmp/qwen-metal-ci.lock")):
            with tempfile.TemporaryDirectory(prefix="qwen-ci-") as directory:
                raw = Path(directory)
                try:
                    run(args, results, raw)
                    results["passed"] = True
                finally:
                    replacements = sorted({v for k, v in os.environ.items() if v and
                                           (k.endswith("_PATH") or k in ("HOME", "QWEN_MLX_FLASH_BIN"))}, key=len, reverse=True)
                    replacements.append(directory)
                    for log in raw.glob("*.log"):
                        content = log.read_text(errors="replace")
                        for value in replacements:
                            content = content.replace(value, "<local-path>")
                        (artifacts / log.name).write_text(content)
    except Exception as error:
        # Exception messages from libraries can contain local paths. Artifacts
        # retain check progress; failure details stay categorical.
        results["error"] = str(error) if type(error) is RuntimeError else type(error).__name__
        print(f"Real Qwen smoke failed: {results['error']}; see sanitized server logs.", flush=True)
    finally:
        (artifacts / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if results["passed"]:
        print("Real Qwen generation checks passed; all owned processes cleaned up.", flush=True)
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
