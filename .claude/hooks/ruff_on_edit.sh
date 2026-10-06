#!/usr/bin/env bash
# PostToolUse hook: after Claude edits a Python file, lint and format-check just that file.
# Problems are returned to Claude on stderr with exit 2 so they get fixed in the same turn.
set -uo pipefail

payload="$(cat)"
path="$(jq -r '.tool_input.file_path // empty' <<<"$payload")"
[[ "$path" == *.py ]] || exit 0
[[ -f "$path" ]] || exit 0

root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
ruff="$root/.venv/bin/ruff"
[[ -x "$ruff" ]] || ruff="$(command -v ruff || true)"
[[ -n "$ruff" ]] || exit 0   # no ruff installed yet (Phase 0); nothing to check

out="$("$ruff" check --quiet "$path" 2>&1; "$ruff" format --check --quiet "$path" 2>&1)"
if [[ -n "$out" ]]; then
  echo "ruff findings in $path (run \`ruff format $path\` / fix the lint):" >&2
  echo "$out" >&2
  exit 2
fi
exit 0
