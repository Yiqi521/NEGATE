#!/usr/bin/env python
"""审计样本：渲染 O6（间隙接受）判"有理由"而回放 C3 判"无理由"的场景（10 s 窗口，专家轨迹 + 冲突车辆轨迹），用于核对实现。
用法: python scripts/render_audit_disagreement.py --negatives results/negatives_navtest_v9.parquet --per-class 10 --out results/audit_disagreement_png
"""
import argparse, os, sys, ast
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from omegaconf import OmegaConf
from hydra.utils import instantiate
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.visualization.plots import plot_bev_frame
from navsim.visualization.bev import add_trajectory_to_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG
from conservative_negatives.definition.objective_criteria import agent_tracks, crossing_conflicts, gap_analysis


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--negatives", type=Path, required=True); ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--split", default="navtest"); a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(a.negatives); sc = d.groupby("token").first().reset_index()
    sc["cls"] = np.where(sc.unprotected_turn, "turn", np.where(sc.merging, "merging", "crossing"))
    # 两个方向的分歧：A = O6 判有理由、回放判无理由；B = 回放判有理由、O6 判无理由
    disA = sc[(~sc.c3_pass.astype(bool)) & sc.c3_replay.astype(bool)].assign(direction="A_gap_justified_replay_not")
    disB = sc[sc.c3_pass.astype(bool) & (~sc.c3_replay.astype(bool))].assign(direction="B_replay_justified_gap_not")
    dis = pd.concat([disA, disB])
    smp = pd.concat([g.sample(min(a.per_class, len(g)), random_state=3) for _, g in dis.groupby(["direction", "cls"])])
    root = Path(os.environ["OPENSCENE_DATA_ROOT"]); dev = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    sf = instantiate(OmegaConf.load(dev / f"navsim/planning/script/config/common/train_test_split/scene_filter/{a.split}.yaml"))
    sf.tokens = smp.token.tolist(); sf.log_names = smp.log_name.unique().tolist(); sf.num_future_frames = 20
    loader = SceneLoader(root / "navsim_logs" / ("test" if a.split == "navtest" else "trainval"), root / "sensor_blobs" / ("test" if a.split == "navtest" else "trainval"), sf, SensorConfig.build_no_sensors())
    cfg_h = TRAJECTORY_CONFIG.get("human", list(TRAJECTORY_CONFIG.values())[0]); rows = []
    for _, r in smp.iterrows():
        if r.token not in loader.tokens: continue
        scene = loader.get_scene_from_token(r.token)
        exp_long = scene.get_future_trajectory(num_trajectory_frames=20).poses.astype(float)
        tracks = agent_tracks(scene); conf = [c for c in crossing_conflicts(exp_long, tracks) if not c["same_stream"]]
        g = gap_analysis(exp_long, tracks, t_c=r.t_c_used)
        fig, ax = plot_bev_frame(scene, 3)
        add_trajectory_to_bev_ax(ax, Trajectory(exp_long[:8].astype(np.float32)), cfg_h)
        # 10 s 专家路径（细线）与冲突车辆的 10 s 轨迹（橙色）+ 冲突点（叉）
        ax.plot(exp_long[:, 1], exp_long[:, 0], color="#2ca02c", lw=0.8, alpha=0.6)
        for c in conf:
            tr = tracks[c["track"]]; ok = ~np.isnan(tr[:, 0])
            ax.plot(tr[ok, 1], tr[ok, 0], color="#ff7f0e", lw=1.2, alpha=0.9)
            ax.plot(c["conflict_xy"][1], c["conflict_xy"][0], "kx", ms=8)
            ax.annotate(f"agent {c['agent_arrival_s']:.1f}s / ego {c['ego_arrival_s']:.1f}s", (c["conflict_xy"][1], c["conflict_xy"][0]), fontsize=6)
        ax.set_title(f"[{r.cls}] {r.token} | t_c={r.t_c_used} stop={r.stop_controlled}\n"
                     f"{r.direction} | gaps={g['gaps_s']} rejected={g['expert_rejected_gaps_s']} acc={g['expert_accepted_gap_s']} | O6 pass={bool(r.c3_pass)} replay pass={bool(r.c3_replay)}", fontsize=7)
        fn = f"{r.direction[:1]}_{r.cls}_{r.token}.png"; fig.savefig(a.out / fn, dpi=110, bbox_inches="tight"); plt.close(fig)
        rows.append(dict(file=fn, direction=r.direction, token=r.token, cls=r.cls, t_c=r.t_c_used, gaps=str(g["gaps_s"]), rejected=str(g["expert_rejected_gaps_s"]),
                         audit_conflict_detection_correct="", audit_gap_reasonable="", note=""))
    pd.DataFrame(rows).to_csv(a.out / "audit_template.csv", index=False); print("rendered", len(rows), "->", a.out)


if __name__ == "__main__":
    main()
