#!/bin/bash
# navtest 相机 blob（32 包，下载约 125 GB，解压约 121 GB）；不下激光雷达（纯相机 LTF 用不到，省 87 GB）。对应手册 §3 的"重新下载"分支。
# 用法: bash ~/navsim_workspace/dataset/download_test_camera_only.sh   （可重复执行；每包解压完成后写标记，重跑跳过）
# 每包先落盘（约 4 GB）再解压，可断点续传；磁盘低于保留线自动停止（MIN_FREE_GB，默认 40）。
source "$(dirname "$(readlink -f "$0")")/_common.sh"
DST="$OPENSCENE_DATA_ROOT/sensor_blobs/test"; mkdir -p "$DST"
for i in $(seq 0 31); do
  mark="$OPENSCENE_DATA_ROOT/sensor_blobs/.test_camera_${i}.done"
  [ -e "$mark" ] && continue
  check_disk 9; log "相机包 $i / 31"
  f="openscene_sensor_test_camera_${i}.tgz"
  fetch "$HF/openscene_sensor_test_camera/$f"
  extract_merge "$DL_TMP/$f" test "$DST"
  rm -f "$DL_TMP/$f"; touch "$mark"
done
log "sensor_blobs/test: $(ls "$DST" | wc -l) 个日志目录（旧机为 147），$(du -sh "$DST" | cut -f1)"
log "TEST CAMERA DONE"
