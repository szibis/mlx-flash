"""Optional handoff to Sentinel's Go-owned lab, inside the shared GPU lock."""

import json
import os
import stat
import subprocess
from contextlib import contextmanager


@contextmanager
def coordinate_lab(environ=None, run=subprocess.run):
    environ = os.environ if environ is None else environ
    values = [environ.get(key, "") for key in (
        "SENTINEL_CI_TOOLS_BIN", "SENTINEL_CI_PROJECT_ROOT", "SENTINEL_CI_LAB_ROOT"
    )]
    if not any(values):
        yield
        return
    if not all(values) or not all(os.path.isabs(value) for value in values):
        raise RuntimeError("Configure all three absolute Sentinel CI lab paths")
    binary, project, root = values
    info = os.lstat(binary)
    if not stat.S_ISREG(info.st_mode) or not os.access(binary, os.X_OK):
        raise RuntimeError("Sentinel CI tools must be an executable regular file")
    command = [binary, "ci-lab"]
    paths = ["--project", project, "--root", root]
    paused = run(command + ["pause", *paths], capture_output=True, text=True, check=True)
    lease = json.loads(paused.stdout)
    if not isinstance(lease, dict) or not isinstance(lease.get("paused"), bool) or not isinstance(lease.get("token"), str):
        raise RuntimeError("Invalid Sentinel lab coordination response")
    if lease["paused"] != bool(lease["token"]):
        raise RuntimeError("Inconsistent Sentinel lab pause ownership")
    try:
        yield
    finally:
        if lease["paused"]:
            run(command + ["resume", *paths, "--token", lease["token"]], capture_output=True, text=True, check=True)
