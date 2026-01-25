#!/usr/bin/env bash
set -euo pipefail

# --------- Config (edit these) ----------
REMOTE_HOST="tempmachine"           # your ssh config Host
REMOTE_DIR="~/lfs"            # remote destination directory
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # project root
# ---------------------------------------

# Default behavior
DRY_RUN=0
DELETE=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/sync_to_remote.sh [--dry-run] [--delete]

Options:
  --dry-run   Show what would be transferred, without changing remote
  --delete    Make remote mirror local (delete remote files not present locally)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --delete)  DELETE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

RSYNC_FLAGS=(-avz --progress)

# Common excludes for code projects; adjust as needed
EXCLUDES=(
  --exclude '.git/'
  --exclude '.vscode/'
  --exclude '.idea/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.DS_Store'
  --exclude 'node_modules/'
  --exclude '.venv/'
  --exclude 'dist/'
  --exclude 'build/'
)

if [[ $DELETE -eq 1 ]]; then
  RSYNC_FLAGS+=(--delete)
fi

if [[ $DRY_RUN -eq 1 ]]; then
  RSYNC_FLAGS+=(--dry-run)
fi

echo "Local : ${LOCAL_DIR}/"
echo "Remote: ${REMOTE_HOST}:${REMOTE_DIR}/"
echo "Args  : ${RSYNC_FLAGS[*]} ${EXCLUDES[*]}"
echo

# Ensure remote dir exists (optional but practical)
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

# Trailing slash on LOCAL_DIR syncs contents into REMOTE_DIR
rsync "${RSYNC_FLAGS[@]}" "${EXCLUDES[@]}" "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
echo "Sync complete."