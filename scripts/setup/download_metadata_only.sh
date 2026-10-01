#!/bin/bash
# 地图 + navtest / navtrain 元数据（下载约 8.5 GB，解压约 17 GB）。对应 docs/minimal_experiment_runbook.md §2。
# 用法: bash ~/navsim_workspace/dataset/download_metadata_only.sh      （可重复执行，已完成的部分跳过）
source "$(dirname "$(readlink -f "$0")")/_common.sh"
cd "$OPENSCENE_DATA_ROOT"

if [ ! -d maps ]; then
  check_disk 3; log "nuPlan 地图（1.0 GB）"
  fetch https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
  rm -rf nuplan-maps-v1.0 && unzip -q "$DL_TMP/nuplan-maps-v1.1.zip" && mv nuplan-maps-v1.0 maps && rm -f "$DL_TMP/nuplan-maps-v1.1.zip"
fi

meta() {  # meta <test|trainval> <解压后约需 GB>
  local s=$1 dst=${1}_navsim_logs
  if [ -d "$dst" ]; then log "$dst 已存在，跳过"; return; fi
  check_disk "$2"; log "元数据 $s"
  fetch "$HF/openscene_metadata_${s}.tgz"
  rm -rf openscene-v1.1 && tar "${TAR_GZ[@]}" -xf "$DL_TMP/openscene_metadata_${s}.tgz"
  mv openscene-v1.1/meta_datas "$dst" && rm -rf openscene-v1.1 "$DL_TMP/openscene_metadata_${s}.tgz"
}
meta test 2
meta trainval 16

# navsim 读 $OPENSCENE_DATA_ROOT/navsim_logs/<split>；兼容压缩包内有 / 没有 <split> 子目录两种布局
mkdir -p navsim_logs
for s in test trainval; do
  if [ -d "${s}_navsim_logs/$s" ]; then ln -sfn "../${s}_navsim_logs/$s" "navsim_logs/$s"; else ln -sfn "../${s}_navsim_logs" "navsim_logs/$s"; fi
done
log "navsim_logs/test: $(ls navsim_logs/test/ | wc -l) 个日志；navsim_logs/trainval: $(ls navsim_logs/trainval/ | wc -l) 个日志；maps: $(ls maps | tr '\n' ' ')"
log "METADATA DONE"
