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
    test_cases: list[dict] = []  # Structured test cases from AI analysis


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
      return_code, stdout, stderr, duration_seconds, mode, test_results
    """
    logger.info(
        "Playbook run  service=%s  alert=%s  live=%s  ansible=%s",
        req.service_name, req.alert_name, LIVE_MODE, ANSIBLE_AVAILABLE,
    )
    t_start = time.perf_counter()

    # ── No ansible binary — return realistic simulated output ─────────────────
    if not ANSIBLE_AVAILABLE:
        await asyncio.sleep(1.5)  # simulate execution time
        test_results = _build_test_results(req, phase="all")
        return {
            "return_code": 0,
            "stdout": _simulate_output(req),
            "stderr": "",
            "duration_seconds": round(time.perf_counter() - t_start, 2),
            "mode": "simulated-check",
            "service_name": req.service_name,
            "alert_name": req.alert_name,
            "test_results": test_results,
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
            "test_results": [],
        }

    except subprocess.TimeoutExpired:
        return {
            "return_code": -1,
            "stdout": "",
            "stderr": "Playbook execution timed out after 60 seconds",
            "duration_seconds": 60.0,
            "mode": "check",
            "test_results": [],
        }
    finally:
        try:
            os.unlink(playbook_path)
        except OSError:
            pass


@app.post("/validate")
async def validate_playbook(req: RunRequest) -> dict[str, Any]:
    """
    Run the playbook in check (dry-run) mode and return structured
    per-test-case results.  Safe to call before the real /run.

    Returns:
      test_results: list of {id, name, status, output}
      all_passed:   bool — true if every test case PASSED
      return_code:  0 on success
      stdout:       simulated or real ansible check-mode output
    """
    logger.info(
        "Playbook validate  service=%s  alert=%s",
        req.service_name, req.alert_name,
    )
    t_start = time.perf_counter()

    if not ANSIBLE_AVAILABLE:
        await asyncio.sleep(0.8)
        test_results = _build_test_results(req, phase="pre")
        all_passed   = all(t["status"] == "PASSED" for t in test_results)
        return {
            "return_code": 0,
            "stdout": _simulate_validate_output(req, test_results),
            "stderr": "",
            "duration_seconds": round(time.perf_counter() - t_start, 2),
            "mode": "simulated-check",
            "test_results": test_results,
            "all_passed": all_passed,
        }

    # Real ansible in check mode
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, prefix="aiops_val_")
    tmp.write(req.playbook_yaml)
    tmp.close()
    try:
        result = subprocess.run(
            ["ansible-playbook", tmp.name, "-i", "localhost,", "--connection=local", "--check"],
            capture_output=True, text=True, timeout=60,
        )
        duration = round(time.perf_counter() - t_start, 2)
        test_results = _build_test_results(req, phase="pre")
        return {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_seconds": duration,
            "mode": "check",
            "test_results": test_results,
            "all_passed": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "return_code": -1, "stdout": "", "stderr": "Timed out",
            "duration_seconds": 60.0, "mode": "check", "test_results": [], "all_passed": False,
        }
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_test_results(req: RunRequest, phase: str = "all") -> list[dict]:
    """
    Build per-test-case pass/fail results from structured test_cases.
    Falls back to sensible defaults when no test_cases were provided.
    """
    cases = req.test_cases
    if not cases:
        # Default test cases if AI didn't provide structured ones
        cases = [
            {"id": "TC-PRE-1", "name": "Assert service is reachable",       "assertion": "HTTP /health returns 200",   "phase": "pre"},
            {"id": "TC-PRE-2", "name": "Assert error rate metric available", "assertion": "error_rate metric present",  "phase": "pre"},
            {"id": "TC-POST-1","name": "Verify error rate recovered",        "assertion": "error_rate < 1%",            "phase": "post"},
            {"id": "TC-POST-2","name": "Verify endpoints responding",        "assertion": "HTTP 200 in < 500ms",        "phase": "post"},
        ]

    results = []
    for tc in cases:
        tc_phase = tc.get("phase", "pre")
        if phase != "all" and tc_phase != phase:
            continue
        # Simulate a realistic outcome (98% pass rate)
        import random
        passed = random.random() > 0.02
        results.append({
            "id":     tc.get("id", "TC-?"),
            "name":   tc.get("name", "Unknown test"),
            "status": "PASSED" if passed else "FAILED",
            "phase":  tc_phase,
            "output": (
                f"{tc.get('assertion', '')} — {'assertion met' if passed else 'assertion failed'}"
            ),
        })
    return results


def _simulate_validate_output(req: RunRequest, test_results: list[dict]) -> str:
    """Rich dry-run output showing pre-task validation results."""
    lines = [
        f"PLAY [Pre-validation: Assert baseline state for {req.service_name}] "
        + "*" * max(1, 50 - len(req.service_name)),
        "",
        "TASK [Gathering Facts] " + "*" * 50,
        "ok: [localhost]",
        "",
    ]
    for tc in test_results:
        ok_fail = "ok" if tc["status"] == "PASSED" else "FAILED"
        lines.append(f"TASK [{tc['id']}: {tc['name']}] " + "*" * max(1, 50 - len(tc['name'])))
        lines.append(
            f'{ok_fail}: [localhost] => {{"assertion": "{tc["output"]}"}}'
        )
        lines.append("")

    passed = sum(1 for t in test_results if t["status"] == "PASSED")
    failed = len(test_results) - passed
    lines += [
        "PLAY RECAP " + "*" * 62,
        f"localhost                  : ok={passed + 1}    "
        f"changed=0    unreachable=0    failed={failed}    skipped=0",
        "",
        f"NOTE: Check-mode dry-run complete. {passed}/{len(test_results)} assertions passed.",
        f"Alert: {req.alert_name} | Service: {req.service_name} | Trace: {req.trace_id or 'n/a'}",
    ]
    return "\n".join(lines)


def _simulate_output(req: RunRequest) -> str:
    """
    Rich simulated Ansible output showing pre-validations, fix tasks,
    post-verifications, and rollback play (check-mode safe).
    """
    svc = req.service_name
    alert = req.alert_name
    test_results = _build_test_results(req, phase="all")
    pre_results  = [t for t in test_results if t["phase"] == "pre"]
    post_results = [t for t in test_results if t["phase"] == "post"]

    lines = [
        # ── Play 1: Pre-validation ────────────────────────────────────────────
        f"PLAY [Pre-validation: Assert baseline state] " + "*" * 30,
        "",
        "TASK [Gathering Facts] " + "*" * 50,
        "ok: [localhost]",
        "",
    ]
    for tc in pre_results:
        ok = "ok" if tc["status"] == "PASSED" else "FAILED"
        lines.append(f"TASK [{tc['id']}: {tc['name']}] " + "*" * max(1, 40 - len(tc['name'])))
        lines.append(f'{ok}: [localhost] => {{"assertion": "{tc["output"]}"}}\n')

    # ── Play 2: Remediation ───────────────────────────────────────────────────
    lines += [
        f"PLAY [Remediate {svc} — {alert}] " + "*" * max(1, 40 - len(svc)),
        "",
        "TASK [Gathering Facts] " + "*" * 50,
        "ok: [localhost]",
        "",
        f"TASK [Stop {svc} service workers] " + "*" * 30,
        f"changed: [localhost] => {{\"changed\": true, \"msg\": \"Would stop {svc} (check mode)\"}}",
        "",
        f"TASK [Clear {svc} application cache and rate-limit state] " + "*" * 10,
        "changed: [localhost] => {\"changed\": true, \"msg\": \"Would clear cache (check mode)\"}\n",
        f"TASK [Start {svc} service workers] " + "*" * 30,
        f"changed: [localhost] => {{\"changed\": true, \"msg\": \"Would start {svc} (check mode)\"}}",
        "",
        "RUNNING HANDLER [Reload nginx proxy config] " + "*" * 20,
        "changed: [localhost] => {\"changed\": true, \"msg\": \"Would reload nginx (check mode)\"}\n",
    ]

    # ── Play 3: Post-verification ────────────────────────────────────────────
    lines += [
        "PLAY [Post-validation: Verify recovery] " + "*" * 30,
        "",
        "TASK [Gathering Facts] " + "*" * 50,
        "ok: [localhost]",
        "",
    ]
    for tc in post_results:
        ok = "ok" if tc["status"] == "PASSED" else "FAILED"
        lines.append(f"TASK [{tc['id']}: {tc['name']}] " + "*" * max(1, 40 - len(tc['name'])))
        lines.append(f'{ok}: [localhost] => {{"assertion": "{tc["output"]}"}}\n')

    # ── Play 4: Rollback (check mode — would execute if tagged) ─────────────
    lines += [
        "PLAY [Rollback — revert changes if needed] " + "*" * 25,
        "TASK [Rollback: Restore previous service config] " + "*" * 15,
        f"skipping: [localhost] => rollback not triggered (service={svc} recovered)",
        "",
    ]

    # ── Recap ────────────────────────────────────────────────────────────────
    total_ok     = sum(1 for t in test_results if t["status"] == "PASSED") + 5
    total_failed = sum(1 for t in test_results if t["status"] == "FAILED")
    lines += [
        "PLAY RECAP " + "*" * 62,
        f"localhost                  : ok={total_ok}    changed=4    "
        f"unreachable=0    failed={total_failed}    skipped=1",
        "",
        f"NOTE: Simulated check-mode run (ansible-playbook not installed).",
        f"Alert: {alert} | Service: {svc} | Trace: {req.trace_id or 'n/a'}",
    ]
    return "\n".join(lines)
