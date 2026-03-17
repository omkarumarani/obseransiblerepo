"""
ansible-runner/app/main.py
────────────────────────────────────────────────────────────────
Ansible playbook execution sidecar for the AIOps Bridge.

Accepts a raw YAML playbook string via POST /run, executes it
(or simulates execution when ansible-playbook is not installed),
and returns the result so the aiops-bridge can post it back to
the xyOps incident ticket.

Safety: runs in --check (dry-run) mode by default.
Set ANSIBLE_LIVE_MODE=true to execute for real (not recommended
in this learning environment — use only with real infrastructure).

Endpoints
─────────
  GET  /health   → {"status":"ok","ansible_available":bool,"live_mode":bool}
  POST /run      → execute or simulate playbook; returns stdout/stderr/rc
────────────────────────────────────────────────────────────────
"""

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ansible-runner")

app = FastAPI(
    title="Ansible Runner",
    description="Executes Ansible playbooks for AIOps automated remediation.",
    version="1.0.0",
)

# ── Config ─────────────────────────────────────────────────────────────────────
# Set ANSIBLE_LIVE_MODE=true to run playbooks for real.
# Default is check/dry-run mode which is safe for a learning environment.
LIVE_MODE: bool = os.getenv("ANSIBLE_LIVE_MODE", "false").lower() == "true"

# Detect at startup whether ansible-playbook binary is present.
ANSIBLE_AVAILABLE: bool = shutil.which("ansible-playbook") is not None


# ── Request model ──────────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    playbook_yaml: str          # Raw YAML content of the Ansible playbook
    service_name: str = "unknown"
    alert_name: str = "unknown"
    trace_id: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {
        "status": "ok",
        "service": "ansible-runner",
        "live_mode": LIVE_MODE,
        "ansible_available": ANSIBLE_AVAILABLE,
        "note": (
            "running real ansible" if ANSIBLE_AVAILABLE
            else "ansible-playbook not installed — using simulated dry-run"
        ),
    }


@app.post("/run")
async def run_playbook(req: RunRequest) -> dict[str, Any]:
    """
    Execute (or dry-run) an Ansible playbook from a YAML string.

    Returns a dict compatible with approval_workflow._execute_playbook():
      return_code, stdout, stderr, duration_seconds, mode
    """
    logger.info(
        "Playbook run  service=%s  alert=%s  live=%s  ansible=%s",
        req.service_name, req.alert_name, LIVE_MODE, ANSIBLE_AVAILABLE,
    )
    t_start = time.perf_counter()

    # ── No ansible binary — return realistic simulated output ─────────────────
    if not ANSIBLE_AVAILABLE:
        await asyncio.sleep(1.5)  # simulate execution time
        return {
            "return_code": 0,
            "stdout": _simulate_output(req),
            "stderr": "",
            "duration_seconds": round(time.perf_counter() - t_start, 2),
            "mode": "simulated-check",
            "service_name": req.service_name,
            "alert_name": req.alert_name,
            "note": "ansible-playbook binary not found — simulated dry-run returned",
        }

    # ── Write playbook to a temp file ──────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, prefix="aiops_"
    )
    tmp.write(req.playbook_yaml)
    tmp.close()
    playbook_path = tmp.name

    try:
        # Run on localhost connection (no remote SSH needed for this lab)
        cmd = [
            "ansible-playbook", playbook_path,
            "-i", "localhost,",
            "--connection=local",
        ]
        if not LIVE_MODE:
            cmd.append("--check")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        duration = round(time.perf_counter() - t_start, 2)
        logger.info(
            "Playbook complete  rc=%d  duration=%.1fs  mode=%s",
            result.returncode, duration, "live" if LIVE_MODE else "check",
        )
        return {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_seconds": duration,
            "mode": "live" if LIVE_MODE else "check",
            "service_name": req.service_name,
            "alert_name": req.alert_name,
        }

    except subprocess.TimeoutExpired:
        return {
            "return_code": -1,
            "stdout": "",
            "stderr": "Playbook execution timed out after 60 seconds",
            "duration_seconds": 60.0,
            "mode": "check",
        }
    finally:
        try:
            os.unlink(playbook_path)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation helper
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_output(req: RunRequest) -> str:
    """
    Return realistic-looking Ansible dry-run output for learning environments
    where the ansible-playbook binary is not installed.
    """
    return (
        f"PLAY [Remediate {req.service_name}] "
        f"{'*' * max(1, 60 - len(req.service_name))}\n\n"
        f"TASK [Gathering Facts] {'*' * 50}\n"
        f"ok: [localhost]\n\n"
        f"TASK [Check service health — {req.alert_name}] {'*' * 30}\n"
        f"ok: [localhost] => {{\"msg\": \"Checking {req.service_name} health metrics\"}}\n\n"
        f"TASK [Restart {req.service_name} workers if error rate > 10%] {'*' * 20}\n"
        f"changed: [localhost] => {{\"changed\": true, \"msg\": "
        f"\"Would restart {req.service_name} (check mode)\"}}\n\n"
        f"TASK [Clear application rate-limit cache] {'*' * 30}\n"
        f"changed: [localhost] => {{\"changed\": true}}\n\n"
        f"TASK [Verify service recovery — wait for error rate < 1%] {'*' * 15}\n"
        f"ok: [localhost]\n\n"
        f"PLAY RECAP {'*' * 62}\n"
        f"localhost                  : "
        f"ok=5    changed=2    unreachable=0    failed=0    skipped=0\n\n"
        f"NOTE: Simulated dry-run (ansible-playbook not installed).\n"
        f"Alert: {req.alert_name} | Service: {req.service_name} | "
        f"Trace: {req.trace_id or 'n/a'}"
    )
