#!/bin/bash
# K-S-CI 过夜流水线：下载（续传）→ sha256 校验 → 流式池化缓存 → HAIL-FFIR 与解析基线
set -u
P=${PYTHON:-python}
D=${AVSIL_KS_ROOT:-external/ks_features}
R=results/ks
URL=https://huggingface.co/datasets/wiberg/AV-CIL_features/resolve/main
mkdir -p $R; rm -f $R/DONE $R/FAILED
declare -A SIZE=([aa]=47185920000 [ab]=47185920000 [ac]=13683185469)
declare -A SHA=([aa]=552ddde617532cd7bf9385e1bd52d41f434d5335f2dbd60e01d3facd7f5c7d6e
                [ab]=14b15b011b5d74105e384e9e950ff4dff59baae0f4ff3be40f19dc81eec2766b
                [ac]=48642aba3bf3bb39d15995b7ff100814bf5d8375f1f491bfa01d198cff7e8e8a)
fail() { echo "FAILED: $*"; echo "$*" > $R/FAILED; exit 1; }
dl() {
  local f=$D/KS_features.zip.$1 n=0
  until [ "$(stat -c%s $f 2>/dev/null || echo 0)" = "${SIZE[$1]}" ]; do
    n=$((n+1)); [ $n -gt 500 ] && fail "download $1"
    curl -sS -L --http1.1 -C - --retry 5 --retry-delay 10 -o $f $URL/KS_features.zip.$1 || sleep 20
  done
  echo "$(date '+%F %T') downloaded $1"
}
echo "$(date '+%F %T') start"
dl aa & dl ab & dl ac & wait
for p in aa ab ac; do
  [ "$(stat -c%s $D/KS_features.zip.$p)" = "${SIZE[$p]}" ] || fail "size $p"
  h=$(sha256sum $D/KS_features.zip.$p | cut -d' ' -f1)
  [ "$h" = "${SHA[$p]}" ] || fail "sha256 $p: $h"
  echo "$(date '+%F %T') sha256 ok $p"
done
cd "$(dirname "$0")/.."
$P ave_cil.py --dataset ks --methods hail hail_exact acil dsal mmal || fail "ave_cil ks"
echo "$(date '+%F %T') done"; touch $R/DONE
