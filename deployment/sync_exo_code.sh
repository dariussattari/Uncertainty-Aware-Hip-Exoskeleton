#!/usr/bin/env bash
# Sync the exo code between the Jetson (~/HipExoGPUCode) and the Mac (../HipExoGPUCode), branch uncertainty-aware-hip-exo.
#   1. On the Jetson: commit any uncommitted changes (git add -A, .gitignore respected), dated by the Mac's clock since
#      the Jetson's is not synced.
#   2. Fetch them to the Mac and fast-forward the Mac's branch; or, if the Mac is ahead, push the Mac's commits to the
#      Jetson (its checkout updates in place). Stops without touching anything if the two have diverged.
# Never pushes to GitHub: run `git -C ../HipExoGPUCode push origin uncertainty-aware-hip-exo` yourself when ready.
# Usage:  deployment/sync_exo_code.sh ["commit message"]
# Env:    EXO_REPO (Mac checkout, default ../HipExoGPUCode), JETSON (ssh host, default jetson),
#         JETSON_REPO (checkout on the Jetson, relative to home, default HipExoGPUCode), BRANCH
set -euo pipefail
cd "$(dirname "$0")/.."

EXO_REPO=${EXO_REPO:-../HipExoGPUCode}
JETSON=${JETSON:-jetson}
JETSON_REPO=${JETSON_REPO:-HipExoGPUCode}
BRANCH=${BRANCH:-uncertainty-aware-hip-exo}
MESSAGE=${1:-"Exo changes from the Jetson"}
MAX_MB=20                                        # refuse to commit files larger than this (logs, data, engines)
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

git -C "$EXO_REPO" rev-parse --abbrev-ref HEAD | grep -qx "$BRANCH" || { echo "Mac checkout is not on $BRANCH" >&2; exit 1; }

# 1. commit on the Jetson
ssh "$JETSON" bash -s -- "$JETSON_REPO" "$BRANCH" "$MAX_MB" "$NOW" "$(printf %s "$MESSAGE" | base64)" <<'REMOTE'
set -euo pipefail
repo=$1; branch=$2; max_mb=$3; now=$4; message=$(printf %s "$5" | base64 -d)
cd ~/"$repo"
[ "$(git rev-parse --abbrev-ref HEAD)" = "$branch" ] || { echo "Jetson checkout is not on $branch" >&2; exit 1; }
git add -A
big=$(git diff --cached --name-only -z | xargs -0 -r du -m 2>/dev/null | awk -v m="$max_mb" '$1 > m {print $2}')
if [ -n "$big" ]; then
    git reset -q
    printf 'not committing; files over %s MB (add them to .gitignore or commit by hand):\n%s\n' "$max_mb" "$big" >&2
    exit 1
fi
if git diff --cached --quiet; then
    echo "Jetson: nothing to commit"
else
    GIT_AUTHOR_DATE=$now GIT_COMMITTER_DATE=$now git commit -q -m "$message"
    git show --stat --format='Jetson committed %h %s' HEAD | head -30
fi
REMOTE

# 2. bring the two branches level
git -C "$EXO_REPO" fetch -q "$JETSON:$JETSON_REPO" "+$BRANCH:refs/remotes/jetson/$BRANCH"
local_sha=$(git -C "$EXO_REPO" rev-parse "$BRANCH")
remote_sha=$(git -C "$EXO_REPO" rev-parse "jetson/$BRANCH")
if [ "$local_sha" = "$remote_sha" ]; then
    echo "in sync at $(git -C "$EXO_REPO" log --oneline -1 "$BRANCH")"
elif git -C "$EXO_REPO" merge-base --is-ancestor "$local_sha" "$remote_sha"; then
    git -C "$EXO_REPO" merge -q --ff-only "jetson/$BRANCH"
    echo "Mac fast-forwarded to $(git -C "$EXO_REPO" log --oneline -1 "$BRANCH")"
elif git -C "$EXO_REPO" merge-base --is-ancestor "$remote_sha" "$local_sha"; then
    git -C "$EXO_REPO" push -q "$JETSON:$JETSON_REPO" "$BRANCH"
    echo "Jetson updated to $(git -C "$EXO_REPO" log --oneline -1 "$BRANCH")"
else
    echo "Mac and Jetson have diverged on $BRANCH; merge by hand: git -C $EXO_REPO merge jetson/$BRANCH" >&2
    exit 1
fi
