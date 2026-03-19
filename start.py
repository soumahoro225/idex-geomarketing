#!/usr/bin/env python3
"""Lancement local — python start.py"""
import subprocess, sys, os
from pathlib import Path

DEPS = open('requirements.txt').read().splitlines()

venv    = Path('.venv')
pip     = str(venv / ('Scripts/pip'     if os.name == 'nt' else 'bin/pip'))
uvicorn = str(venv / ('Scripts/uvicorn' if os.name == 'nt' else 'bin/uvicorn'))

if not venv.exists():
    subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)

subprocess.run([pip, 'install', '--quiet', '--upgrade', 'pip'], check=True)
subprocess.run([pip, 'install', '--quiet'] + DEPS, check=True)

print('\n  \033[92m▶  http://localhost:8000\033[0m  (Ctrl+C pour arrêter)\n')
os.execv(uvicorn, [uvicorn, 'server:app', '--reload', '--port', '8000'])
