#!/usr/bin/env bash
set -e
cd /home/JJ_Group/lih2511/test/jev || exit 1
git add -A
if ! git diff --cached --quiet; then
  git commit -m "Sync jev $(date '+%Y-%m-%d %H:%M:%S')"
fi
git push --quiet origin main
