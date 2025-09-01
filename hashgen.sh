#!/usr/bin/env bash

# hashgen.sh
#
# This script runs the generate_hash.py utility for generating bcrypt password
# hashes. It will attempt to locate a Python interpreter in a virtual
# environment (./venv) if one exists; otherwise it falls back to the
# system Python. To use, run:
#   ./hashgen.sh
# or make the script executable first: chmod +x hashgen.sh

# Navigate to the directory where this script resides so relative paths work
cd "$(dirname "$0")"

# Default to python3, but fall back to python if python3 isn't available
PY="python3"
if [ -x "venv/bin/python" ]; then
  PY="venv/bin/python"
elif [ -x "venv/Scripts/python.exe" ]; then
  # In case the script is run in WSL or a similar environment where Windows
  # venv structure is present. It may still work under WSL bash.
  PY="venv/Scripts/python.exe"
elif ! command -v python3 >/dev/null 2>&1 && command -v python >/dev/null 2>&1; then
  PY="python"
fi

exec "$PY" generate_hash.py