#!/usr/bin/env bash
# Assemble a clean folder to push to a Hugging Face Space.
# Usage: deploy/huggingface/prepare_space.sh <target-dir>
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
target="${1:?usage: prepare_space.sh <target-dir>}"

mkdir -p "$target/models"
cp "$here/Dockerfile" "$target/Dockerfile"
cp "$here/README.md" "$target/README.md"

# Never ship the local database (it holds real accounts and scans) or caches.
rsync -a --delete --exclude '__pycache__' --exclude '*.db' --exclude '*.db-*' "$root/backend/" "$target/backend/"
rsync -a --delete --exclude '__pycache__' "$root/ml_pipeline/" "$target/ml_pipeline/"

cp "$root/models/skinscan_densenet121_v4.pth" "$target/models/"

# Spaces require Git LFS for files over 10 MB.
printf '*.pth filter=lfs diff=lfs merge=lfs -text\n' > "$target/.gitattributes"

echo "Space folder ready: $target"
du -sh "$target"
