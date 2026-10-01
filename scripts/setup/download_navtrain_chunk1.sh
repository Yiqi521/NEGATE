#!/bin/bash
# navtrain 第 1 个当前帧压缩包（下载 54.5 GB，解压约 75 GB，约 300 个日志）。对应手册 §4。
# 用法: bash ~/navsim_workspace/dataset/download_navtrain_chunk1.sh
#       tail -2 ~/navsim_workspace/dataset/download_navtrain_chunk1.log      # 最后一行 ALL DONE 即完成
# 默认：断点续传下载 → 官方 MD5 校验（NAVSIM docs/splits.md）→ 解压合并 → 删包。峰值多占 55 GB；
#       WSL 下建议 DL_TMP=/mnt/c/Users/<用户名>/Downloads/navsim_dl，压缩包放在 Windows 盘，删后空间立即归还。
# STREAM=1：边下边解压、不落盘（旧机做法）；中途断网需整包重下，也无法事先校验 MD5。
source "$(dirname "$(readlink -f "$0")")/_common.sh"
exec > >(tee -a "$OPENSCENE_DATA_ROOT/download_navtrain_chunk1.log") 2>&1
URL=https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_current_1.tgz
MD5=6f92f38d5f03ed852da7872a7122bdd2
DST="$OPENSCENE_DATA_ROOT/sensor_blobs/trainval"; mark="$OPENSCENE_DATA_ROOT/sensor_blobs/.navtrain_current_1.done"
if [ -e "$mark" ]; then log "已完成（$mark）"; log "ALL DONE"; exit 0; fi

if [ "${STREAM:-0}" = 1 ]; then
  check_disk 80; log "流式下载 + 解压 navtrain_current_1（54.5 GB）"
  tmp="${OPENSCENE_DATA_ROOT:?}/_x_navtrain_current_1"; rm -rf "${tmp:?}"; mkdir -p "$tmp"
  curl -fsSL --retry 10 "$URL" | tee >(md5sum | cut -d' ' -f1 > "$tmp.md5") | tar "${TAR_GZ[@]}" -xf - -C "$tmp"
  sleep 2; got=$(cat "$tmp.md5"); rm -f "$tmp.md5"
  [ "$got" = "$MD5" ] || { log "MD5 不符（$got ≠ $MD5）：数据可能缺文件，请删除 $tmp 后重下"; exit 4; }
  src=$(find "$tmp" -maxdepth 2 -type d -name 'current_split_1' -print -quit); [ -n "$src" ] || src="$tmp"
  mkdir -p "$DST" && cp -rlf "$src"/. "${DST:?}"/ && rm -rf "${tmp:?}"
else
  check_disk 135; log "下载 navtrain_current_1.tgz（54.5 GB，可断点续传）到 $DL_TMP"
  fetch "$URL"
  log "校验 MD5"; echo "$MD5  $DL_TMP/navtrain_current_1.tgz" | md5sum -c -
  log "解压"; extract_merge "$DL_TMP/navtrain_current_1.tgz" current_split_1 "$DST"
  rm -f "$DL_TMP/navtrain_current_1.tgz"
fi
touch "$mark"
log "sensor_blobs/trainval: $(ls "$DST" | wc -l) 个日志目录（期望约 300），$(du -sh "$DST" | cut -f1)"
log "ALL DONE"
