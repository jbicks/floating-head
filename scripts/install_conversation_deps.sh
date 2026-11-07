#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "Python venv support is missing. Install python3-venv, then retry." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
if .venv/bin/python -m pip show opencv-python-headless >/dev/null 2>&1; then
  .venv/bin/python -m pip uninstall -y opencv-python-headless
fi
.venv/bin/python -m pip install -r requirements-conversation.txt
printf '\nVisitor capture dependencies are ready.\n'
printf 'Person detection is an optional external plugin; see README.md to configure one.\n'
printf './run.sh will use .venv automatically.\n'
