#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Keep diagnostic logs private on the local machine.
umask 077
mkdir -p logs
chmod 700 logs
log_file="$PWD/logs/headroom.log"
printf '\n===== %s ./run.sh %s =====\n' "$(date -Is)" "$*" >>"$log_file"
chmod 600 "$log_file"

# Prefer Pi's stored OpenAI key, matching the opt-out behavior of the separate
# Halloween app. The key is inherited only by this server process; it is never
# written into this project or sent to the browser.
credential_source="environment"
if [[ "${OPENAI_USE_PI_AUTH:-1}" == "1" ]]; then
  pi_auth_file="$HOME/.pi/agent/auth.json"
  if [[ -r "$pi_auth_file" ]]; then
    pi_auth_key="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("openai", {}).get("key", ""))' "$pi_auth_file" 2>/dev/null || true)"
    if [[ -n "$pi_auth_key" ]]; then
      export OPENAI_API_KEY="$pi_auth_key"
      credential_source="Pi"
    fi
    unset pi_auth_key
  fi
fi
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo "Using OpenAI credential from $credential_source."
  printf '%s OpenAI credential source: %s (key value not logged)\n' "$(date -Is)" "$credential_source" >>"$log_file"
else
  echo "No OpenAI key found; the local server will start, but speech is unavailable." >&2
  printf '%s OpenAI credential unavailable\n' "$(date -Is)" >>"$log_file"
fi

# WSL needs to listen on its virtual interface for the Windows port proxy to
# relay the iPad connection. Keep native Linux/Windows-style local runs loopback
# only unless the caller explicitly supplies HOST.
if [[ -z "${HOST:-}" ]]; then
  if [[ -n "${WSL_DISTRO_NAME:-}" ]] || grep -qiE 'microsoft|wsl' /proc/sys/kernel/osrelease 2>/dev/null; then
    export HOST="0.0.0.0"
  else
    export HOST="127.0.0.1"
  fi
fi
export PORT="${PORT:-8766}"

# Replace a stale Headroom instance instead of failing with EADDRINUSE. Only
# stop a listener that we can verify belongs to this project; never kill an
# unrelated service just because it shares the configured port.
if command -v ss >/dev/null 2>&1; then
  listener_output="$(ss -H -ltnp "sport = :$PORT" 2>/dev/null || true)"
else
  # Keep fresh launches usable on minimal Linux hosts; without ss we can test
  # whether the port is free but cannot safely identify/restart an owner.
  if python3 -c 'import socket,sys; s=socket.socket(); s.bind((sys.argv[1],int(sys.argv[2]))); s.close()' "$HOST" "$PORT" >/dev/null 2>&1; then
    listener_output=""
  else
    echo "Port $PORT is in use, but ss is unavailable to identify its owner; refusing to stop it." >&2
    exit 1
  fi
fi
if [[ -n "$listener_output" ]]; then
  mapfile -t listener_pids < <(grep -oE 'pid=[0-9]+' <<<"$listener_output" | cut -d= -f2 | sort -u)
  if [[ ${#listener_pids[@]} -eq 0 ]]; then
    echo "Port $PORT is already in use, but its owner cannot be identified; refusing to stop it." >&2
    exit 1
  fi

  project_root="$(pwd -P)"
  for pid in "${listener_pids[@]}"; do
    [[ -r "/proc/$pid/cmdline" ]] || continue
    pid_cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
    pid_cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    if [[ "$pid_cmdline" != *"$project_root/server.py"* ]] \
      && ! { [[ "$pid_cwd" == "$project_root" ]] && [[ " $pid_cmdline " == *" server.py "* ]]; }; then
      pid_name="$(cat "/proc/$pid/comm" 2>/dev/null || echo unknown)"
      echo "Port $PORT is already in use by $pid_name (PID $pid), not this Headroom project; refusing to stop it." >&2
      exit 1
    fi
  done

  echo "Stopping existing Headroom server on port $PORT gracefully..."
  for pid in "${listener_pids[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  # Wait for the process itself as well as its listening socket: server_close()
  # drains active HTTP requests, including Rhubarb subprocesses, before exit.
  stopped=0
  for _ in $(seq 1 1800); do
    listener_output="$(ss -H -ltnp "sport = :$PORT" 2>/dev/null || true)"
    still_running=0
    for pid in "${listener_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        process_state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
        [[ "$process_state" == "Z" ]] || still_running=1
      fi
    done
    if [[ -z "$listener_output" && "$still_running" -eq 0 ]]; then
      stopped=1
      break
    fi
    sleep 0.1
  done
  if [[ "$stopped" -ne 1 ]]; then
    echo "Headroom is still shutting down on port $PORT. Not force-killing it, to avoid leaving active work or child processes behind; retry ./run.sh after it exits." >&2
    exit 1
  fi
fi

python_bin="python3"
if [[ -x .venv/bin/python ]]; then
  python_bin=".venv/bin/python"
fi
exec "$python_bin" server.py
