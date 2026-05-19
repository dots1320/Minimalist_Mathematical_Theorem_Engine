#!/usr/bin/env bash
# macOS convenience launcher for the math theorem correction chat.
# Requires: Python 3.10+, Apple Silicon (M1/M2/M3), dependencies installed via:
#   pip install -r requirements/requirements.txt
#   pip install -r requirements/requirements-mac.txt

cd "$(dirname "$0")"
python scripts/inference/chat_mac.py "$@"
