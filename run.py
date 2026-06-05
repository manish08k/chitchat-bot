"""
run.py — Definitive launcher.
Run once: /usr/local/bin/python3 run.py
It installs the project as a package into the venv (pip install -e .)
so ALL submodules are importable by any Python subprocess, always.
"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Find venv Python ──────────────────────────
venv_python = None
for candidate in [
    PROJECT_ROOT / "venv" / "bin" / "python3.11",
    PROJECT_ROOT / "venv" / "bin" / "python3",
    PROJECT_ROOT / "venv" / "bin" / "python",
]:
    if candidate.exists():
        venv_python = str(candidate)
        break

if not venv_python:
    print("❌ No venv found.")
    print("   Run: python3.11 -m venv venv && venv/bin/pip install -r requirements.txt")
    sys.exit(1)

print(f"✅ Using Python: {venv_python}")

# ── Install project as editable package ───────
# This writes an egg-link / direct_url into the venv's site-packages
# that points to PROJECT_ROOT. Every Python using this venv — including
# uvicorn's reload subprocesses — will find api, core, etc. as top-level packages.
print("📦 Installing project into venv (pip install -e .) ...")
result = subprocess.run(
    [venv_python, "-m", "pip", "install", "-e", ".", "--quiet"],
    cwd=str(PROJECT_ROOT),
)
if result.returncode != 0:
    print("❌ pip install -e . failed. Check output above.")
    sys.exit(1)

print("✅ Project installed as editable package.")
print(f"▶  API docs → http://localhost:8000/docs\n")

# ── Launch uvicorn using the VENV python directly ─
cmd = [
    venv_python, "-m", "uvicorn", "main:app",
    "--reload", "--host", "0.0.0.0", "--port", "8000"
]

os.execv(venv_python, cmd)   # replace this process — no subprocess, no env loss