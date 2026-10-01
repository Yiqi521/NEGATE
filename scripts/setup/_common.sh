# download_*.sh 共用：环境变量、磁盘保护、可续传下载、解压合并。由各脚本 source，不单独执行。
set -euo pipefail
# shellcheck disable=SC1091
source "$HOME/navsim_workspace/env.sh"          # 由 bootstrap_wsl.sh 生成
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1
MIN_FREE_GB=${MIN_FREE_GB:-40}                   # 任何一步开始前至少保留的空间
DL_TMP=${DL_TMP:-$OPENSCENE_DATA_ROOT/_dl}        # 压缩包临时目录；WSL 下可设为 /mnt/c/... 以免虚拟盘膨胀（虚拟盘删文件后不自动缩小）
mkdir -p "$DL_TMP"
command -v pigz >/dev/null && TAR_GZ=(-I pigz) || TAR_GZ=(-z)

log() { echo "[$(date '+%F %T')] $*"; }

free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }

# WSL 里 ext4 的 df 显示的是虚拟盘上限（默认 1 TB），真正的约束是宿主 C: 盘，所以两者取小
check_disk() {  # check_disk <本步需要的 GB>
  local need=${1:-0} f fc
  f=$(free_gb "$OPENSCENE_DATA_ROOT")
  if [ -d /mnt/c ]; then fc=$(free_gb /mnt/c); [ "$fc" -lt "$f" ] && f=$fc; fi
  if [ $((f - need)) -lt "$MIN_FREE_GB" ]; then
    log "磁盘剩余 ${f} GB，本步约需 ${need} GB，会低于保留线 ${MIN_FREE_GB} GB，停止（可调 MIN_FREE_GB）"; exit 2
  fi
}

fetch() {  # fetch <url>：下载到 $DL_TMP/<文件名>，可断点续传；已完整时 wget 直接返回
  wget -c --tries=30 --retry-connrefused --waitretry=15 --progress=dot:giga -P "$DL_TMP" "$1"
}

# extract_merge <tgz> <锚点目录名> <目标目录>
#   把压缩包解到同盘临时目录，找到名为 <锚点> 的目录（如 test、current_split_1），用硬链接合并进目标目录后删除临时目录。
#   不依赖压缩包内的前缀层数，同一日志被拆在多个包里时也能正确合并。
extract_merge() {
  local tgz=$1 anchor=$2 dst=$3 tmp src
  tmp="${OPENSCENE_DATA_ROOT:?}/_x_$(basename "$tgz" .tgz)"
  rm -rf "${tmp:?}"; mkdir -p "$tmp" "$dst"
  tar "${TAR_GZ[@]}" -xf "$tgz" -C "$tmp"
  src=$(find "$tmp" -maxdepth 4 -type d -name "$anchor" -print -quit)   # -quit 而非 | head：避免 pipefail 下 SIGPIPE 误报失败
  if [ -z "$src" ]; then log "在 $tgz 里找不到目录 $anchor；解压结果保留在 $tmp"; find "$tmp" -maxdepth 3 | sed -n 1,10p; exit 3; fi
  cp -rlf "$src"/. "${dst:?}"/ && rm -rf "${tmp:?}"                     # -f：重跑时覆盖已存在的同名文件
}
