#!/usr/bin/env bash
cd /home/JJ_Group/lih2511/test/jev || exit 1
exec 9>/tmp/jev-github-sync.lock
flock -n 9 || exit 0

while true; do
  if git add -A && ! git diff --cached --quiet; then
    git commit -m "Sync jev $(date '+%Y-%m-%d %H:%M:%S')"
  fi
  git push --quiet origin main
  sleep 30
done
