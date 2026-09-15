"""实验追踪（研究计划 §13）：零依赖的本地记录 + 可选镜像到 TensorBoard / W&B / MLflow。

设计取舍：navsim 环境把 numpy 锁在 1.23.4，安装 mlflow / wandb 有破坏环境的风险，
因此核心实现不依赖任何第三方库，只写 JSON / JSONL / CSV；若目标机器已装 TensorBoard
（本机已有）或 W&B / MLflow，自动镜像，无需改调用代码。

每个 run 落盘到 <root>/<run_id>/：
    meta.json     运行元信息：名称、时间、git commit 与是否 dirty、conda 环境、GPU、完整配置快照
    metrics.jsonl 逐步指标，一行一条 {"step":…, "wall_s":…, …}
    summary.json  最终指标（run.finish(...) 写入）
    files/        额外落盘的产物（配置 yaml、结果 csv 等）
并在 <root>/index.csv 追加一行，便于横向比较。

用法:
    from conservative_negatives.tracking import Run
    with Run("ltf_lambda0.1_seed0", config={"lambda_neg": 0.1, "seed": 0}, tags=["step4"]) as run:
        for step, ... in enumerate(...):
            run.log({"loss": l, "loss_neg": ln, "neg_active_frac": f}, step=step)
        run.finish({"pdms": 0.845, "ep": 0.79})
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_ROOT = Path(os.environ.get("CN_RUNS_ROOT", "results/runs"))
INDEX_COLUMNS = ["run_id", "name", "started", "finished", "duration_s", "git_commit", "git_dirty", "tags", "status"]


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "host": platform.node(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "navsim_devkit_root": os.environ.get("NAVSIM_DEVKIT_ROOT"),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:
        pass
    return info


def _jsonable(o: Any) -> Any:
    """把 numpy / torch / Path 等转成可序列化对象。"""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    for attr in ("item", "tolist"):
        if hasattr(o, attr) and not isinstance(o, (str, bytes)):
            try:
                return getattr(o, attr)()
            except Exception:
                pass
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


class Run:
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None,
                 root: Optional[Path] = None, tensorboard: bool = True, wandb_project: Optional[str] = None,
                 mlflow_uri: Optional[str] = None):
        self.name = name
        self.root = Path(root or DEFAULT_ROOT)
        self.run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
        self.dir = self.root / self.run_id
        (self.dir / "files").mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.tags = tags or []
        self._metrics_fh = open(self.dir / "metrics.jsonl", "a", buffering=1)
        self._status = "running"

        commit, dirty = _git("rev-parse", "HEAD"), _git("status", "--porcelain")
        self.meta = {
            "run_id": self.run_id, "name": name, "started": datetime.now().isoformat(timespec="seconds"),
            "tags": self.tags, "git_commit": commit, "git_dirty": bool(dirty), "git_dirty_files": len((dirty or "").splitlines()),
            "env": _env_info(), "config": _jsonable(config or {}), "argv": " ".join(os.sys.argv),
        }
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False))

        # 可选镜像
        self._tb = self._wandb = self._mlflow = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(str(self.dir / "tb"))
            except Exception:
                pass
        if wandb_project:
            try:
                import wandb
                self._wandb = wandb.init(project=wandb_project, name=name, config=config, tags=self.tags, reinit=True)
            except Exception:
                pass
        if mlflow_uri:
            try:
                import mlflow
                mlflow.set_tracking_uri(mlflow_uri); mlflow.start_run(run_name=name)
                if config:
                    mlflow.log_params({k: str(v) for k, v in config.items()})
                self._mlflow = mlflow
            except Exception:
                pass
        print(f"[tracking] run {self.run_id} -> {self.dir}"
              f"{' (+tensorboard)' if self._tb else ''}{' (+wandb)' if self._wandb else ''}{' (+mlflow)' if self._mlflow else ''}")

    # ---------- logging ----------
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        rec = {"step": step, "wall_s": round(time.time() - self.t0, 3), **_jsonable(metrics)}
        self._metrics_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if self._tb:
                    self._tb.add_scalar(k, float(v), step or 0)
                if self._mlflow:
                    try:
                        self._mlflow.log_metric(k, float(v), step=step or 0)
                    except Exception:
                        pass
        if self._wandb:
            try:
                self._wandb.log(_jsonable(metrics), step=step)
            except Exception:
                pass

    def log_artifact(self, path: Path, name: Optional[str] = None) -> Path:
        """把文件复制进 run 目录（配置快照、结果表等）。"""
        import shutil
        dst = self.dir / "files" / (name or Path(path).name)
        shutil.copy2(path, dst)
        return dst

    def save_text(self, name: str, text: str) -> Path:
        dst = self.dir / "files" / name
        dst.write_text(text, encoding="utf-8")
        return dst

    # ---------- finish ----------
    def finish(self, summary: Optional[Dict[str, Any]] = None, status: str = "finished") -> None:
        self._status = status
        s = _jsonable(summary or {})
        s.update({"duration_s": round(time.time() - self.t0, 1), "status": status,
                  "finished": datetime.now().isoformat(timespec="seconds")})
        (self.dir / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
        self._metrics_fh.close()
        if self._tb:
            self._tb.close()
        if self._wandb:
            try:
                self._wandb.summary.update(s); self._wandb.finish()
            except Exception:
                pass
        if self._mlflow:
            try:
                for k, v in s.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        self._mlflow.log_metric(k, float(v))
                self._mlflow.end_run()
            except Exception:
                pass
        self._append_index(s)
        print(f"[tracking] {self.run_id} {status} ({s['duration_s']}s)")

    def _append_index(self, summary: Dict[str, Any]) -> None:
        idx = self.root / "index.csv"
        row = {"run_id": self.run_id, "name": self.name, "started": self.meta["started"],
               "finished": summary.get("finished"), "duration_s": summary.get("duration_s"),
               "git_commit": (self.meta["git_commit"] or "")[:8], "git_dirty": self.meta["git_dirty"],
               "tags": ";".join(self.tags), "status": summary.get("status")}
        extra = {k: v for k, v in summary.items() if k not in row and isinstance(v, (int, float, str, bool))}
        row.update(extra)
        exists = idx.exists()
        prev_cols: List[str] = []
        if exists:
            with open(idx) as f:
                prev_cols = next(csv.reader(f), [])
        cols = prev_cols or INDEX_COLUMNS
        for k in row:
            if k not in cols:
                cols.append(k)
        rows = []
        if exists:
            with open(idx) as f:
                rows = list(csv.DictReader(f))
        rows.append({k: row.get(k, "") for k in cols})
        with open(idx, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})

    # ---------- context manager ----------
    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._status == "running":
            self.finish(status="failed" if exc_type else "finished")


def list_runs(root: Optional[Path] = None):
    """读取 index.csv 为 DataFrame（需要 pandas）。"""
    import pandas as pd
    idx = Path(root or DEFAULT_ROOT) / "index.csv"
    return pd.read_csv(idx) if idx.exists() else pd.DataFrame(columns=INDEX_COLUMNS)
