#!/usr/bin/env bash
# swap_base_path.sh  <old_prefix>  <new_prefix>
# Example:  ./swap_base_path.sh "/home2/ebrahim" "/home3/ebrahim2"

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <old_prefix> <new_prefix>" >&2
  exit 1
fi

OLD=$1
NEW=$2

git grep -IlZ "$OLD" -- . ':(exclude).git' \
  | xargs -0 sed -i -e "s|$OLD|$NEW|g"

echo "Done!  Replaced every ‘$OLD’ with ‘$NEW’.  Review with ‘git diff’."

