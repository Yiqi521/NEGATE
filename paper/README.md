# 论文源文件（IEEE 会议格式）

```
paper/
├── main.tex              正文（IEEEtran, conference, 双栏）
├── refs.bib              参考文献（条目已逐条核对过出处）
├── figs/make_figures.py  生成 4 张数据图的脚本
└── figs/*.pdf            脚本产物，被 main.tex 直接 \includegraphics
```

## 编译

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`IEEEtran.cls` 与 `IEEEtran.bst` 随 TeX Live / Overleaf 提供，仓库里不需要额外文件。
本机没有 TeX，验证用的是隔离 conda 环境里的 tectonic：

```bash
conda activate texbuild
tectonic -X compile main.tex        # 已通过，7 页，无 error、无 overfull box
```

## 图是怎么画的

分两类，都是矢量，没有一张位图。

**数据图（图 3–6）** 由 `figs/make_figures.py` 从 `results/` 下的文件直接读数生成，
脚本里没有手抄的数字，标签更新后重跑一次即可刷新论文。约定：

- 画布宽度固定 `3.45 in`（IEEE 单栏 249 pt），`savefig.bbox="standard"` +
  `constrained_layout`，保证导出宽度正好等于栏宽。**不要用 `bbox_inches="tight"`**——
  文字超出画布时它会把图撑宽，插进 LaTeX 后就会压到栏间空白。
- 正文字号 8 pt、刻度 7.5 pt，与图注字号相当；图按 1:1 放入页面，不做任何缩放，
  这样最小字号不会掉到 IEEE 要求的 8 pt 以下。
- 衬线字体用 Nimbus Roman（与 Times 度量兼容），`pdf.fonttype=42` 嵌入真实字形。
- 配色只用三个色相（蓝 `#2A78D6` / 橙 `#EB6834` / 青 `#1BAF7A`），已做色觉分离校验；
  柱子另加网格线纹理，黑白打印仍可区分。
- 每根柱子直接标数值，不依赖读者去对刻度；两组以上一定有图例。

**示意图（图 1 流程、图 2 几何）** 直接写在 `main.tex` 里的 TikZ，
好处是字体与正文一致、随文档一起缩放、不产生外部依赖。
图 2 的坐标是按真实遮挡几何算的：视线从自车经停放车辆的角点切出去，
交到冲突车道上得到可见极限点，幻影车画在该点之外。

## 尚未完成、文中已标注的部分

正文第 VII 节整节是训练部分，用 `\wip`（橙色星号）标记；
图 1 最右侧的"separation loss"框为虚线框，同样标了 in progress。
