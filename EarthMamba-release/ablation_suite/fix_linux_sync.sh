#!/usr/bin/env bash
# Deprecated: use  python3 ablation_study/run/fix_linux_sync.py
exec python3 "$(dirname "$0")/fix_linux_sync.py" "$@"
