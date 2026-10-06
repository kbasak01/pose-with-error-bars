#!/usr/bin/env bash
# PreToolUse guard for pose-with-error-bars.
# Blocks (exit 2, reason on stderr) any tool call that would:
#   - edit/write inside the read-only P1 submodule (external/)
#   - edit/write inside the SPEED+ dataset tree or a speedplus* path
#   - run a shell command that commits/moves the submodule, deletes data, or re-downloads SPEED+
# Everything else passes through to the normal permission flow (exit 0).
set -euo pipefail

payload="$(cat)"
tool="$(jq -r '.tool_name // empty' <<<"$payload")"

block() {
  echo "BLOCKED by .claude/hooks/guard_paths.sh: $1. See CLAUDE.md invariants 1 and 9." >&2
  exit 2
}

case "$tool" in
  Edit|Write|MultiEdit|NotebookEdit)
    path="$(jq -r '.tool_input.file_path // .tool_input.notebook_path // empty' <<<"$payload")"
    [[ -z "$path" ]] && exit 0
    case "$path" in
      */external/*|external/*) block "P1 submodule is read-only ($path)" ;;
      */datasets/*|*/speedplusv2/*|*/speedplus/*|*/data/speedplus) block "dataset tree is read-only ($path)" ;;
    esac
    ;;
  Bash)
    cmd="$(jq -r '.tool_input.command // empty' <<<"$payload")"
    if grep -Eq 'git +-C +external[^ ]* +(commit|checkout|reset|pull|push|rebase|merge|switch|restore|clean)' <<<"$cmd"; then
      block "git write operation inside the P1 submodule"
    fi
    if grep -Eq 'cd +external[^;&|]*(;|&&)[^;&|]*git +(commit|checkout|reset|pull|push|rebase|merge|switch|restore|clean)' <<<"$cmd"; then
      block "git write operation inside the P1 submodule"
    fi
    if grep -Eq '(^|[;&| ])(rm|mv|rsync|truncate)( [^;&|]*)? [^;&|]*(external/|datasets/|speedplus(v2)?(/|$| ))' <<<"$cmd"; then
      block "destructive command touching the submodule or dataset"
    fi
    if grep -Eqi '(wget|curl|aria2c|gdown)[^;&|]*(speedplus|purl\.stanford\.edu|stacks\.stanford\.edu)' <<<"$cmd"; then
      block "SPEED+ is already on disk; never re-download it"
    fi
    if grep -Eq 'git +submodule +(update +--remote|deinit|set-branch)' <<<"$cmd"; then
      block "changing the P1 submodule pin needs a DECISIONS.md entry and a parity re-run; ask the user"
    fi
    ;;
esac
exit 0
