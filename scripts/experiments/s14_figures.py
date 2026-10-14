"""Paper figures (static PNG, Chinese labels). Output: outputs/figures/."""

import sys

import _common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from pickleball_experiment.config import SEL_P, SEL_U  # noqa: E402
from pickleball_experiment.dataset import load_clips  # noqa: E402
from pickleball_experiment.kinematics import IDX, arms, elbow_angle, mid, phases  # noqa: E402
from pickleball_experiment.paths import work  # noqa: E402
from pickleball_experiment.restoration import interp_restore, smooth_restore  # noqa: E402

C = load_clips()
OUT = work("figures")
OUT.mkdir(exist_ok=True)
LAM = SEL_P["lambda"]
A1, A3 = SEL_P["a_backswing"], SEL_P["a_followthrough"]
for f in fm.findSystemFonts():
    if "NotoSansCJK-Regular" in f or "NotoSansCJK-Bold" in f:
        fm.fontManager.addfont(f)
plt.rcParams.update(
    {
        "font.family": [
            "Noto Sans CJK JP",
            "Noto Sans CJK SC",
            "Microsoft YaHei",
            "SimHei",
            "PingFang SC",
        ],
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8984",
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.grid": True,
        "grid.color": "#ecebe7",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.dpi": 300,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    }
)
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8984", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COL = {"linear": "#c3c2b7", "pchip": "#8a8984", "U": ORANGE, "P": BLUE}
LAB = {
    "linear": "线性插值",
    "pchip": "PCHIP",
    "U": "统一平滑修复（U）",
    "P": "阶段感知平滑修复（P）",
}
PH = {1: ("引拍", "#eef3fb"), 2: ("前挥/触球", "#d6e4f7"), 3: ("随挥", "#eef3fb")}
EDGES = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
]
X1 = "clip_000081"


def skel(ax, c, fi, dx, arm, sc=1.0, occ=None):
    xy = c["xy"][fi].copy()
    hip = (xy[11] + xy[12]) / 2
    xy = (xy - hip) * sc
    xy[:, 0] += dx
    hit = {"right": {6, 8, 10}, "left": {5, 7, 9}}[arm]
    for a, b in EDGES:
        h = a in hit and b in hit
        dashed = occ is not None and (occ.get(a) or occ.get(b)) and h
        ax.plot(
            [xy[a, 0], xy[b, 0]],
            [-xy[a, 1], -xy[b, 1]],
            color=BLUE if h else "#9d9c96",
            lw=2.2 if h else 1.3,
            ls=(0, (2, 1.5)) if dashed else "-",
            solid_capstyle="round",
            zorder=3 if h else 2,
        )
    for k in [5, 6, 7, 8, 9, 10, 11, 12]:
        h = k in hit
        o = occ is not None and occ.get(k)
        ax.scatter(
            xy[k, 0],
            -xy[k, 1],
            s=14 if h else 7,
            color="white" if o else (BLUE if h else "#9d9c96"),
            edgecolor=BLUE if h else "#9d9c96",
            lw=1,
            zorder=4,
        )
    return xy


# ======================= 图1 实验流程图 =======================
def fig1():
    c = C[X1]
    ph = phases(c)
    arm = c["final"]["hitting_arm"]
    fig = plt.figure(figsize=(7.4, 4.5))
    W = 0.2
    G = 0.05
    L0 = 0.015
    T = 0.60
    H = 0.30
    pos = [[L0 + k * (W + G), T, W, H] for k in range(4)]
    titles = [
        "① 比赛视频与二维姿态",
        "② 人工标注阶段与可见性",
        "③ 合成遮挡与轨迹修复",
        "④ 二维肘角误差（MAE，°）",
    ]
    axs = [fig.add_axes(p) for p in pos]
    for k in range(4):
        fig.text(pos[k][0], T + H + 0.06, titles[k], fontsize=8, weight="bold")
    # panel 1
    ax = axs[0]
    ax.axis("off")
    F = int(np.where(ph == 2)[0].min())
    Cc = int(np.where(ph == 2)[0].max())
    for k, fi in enumerate([max(F - 7, 0), F, Cc, min(Cc + 6, c["N"] - 1)]):
        skel(ax, c, fi, k * 78, arm, 0.45)
    ax.set_xlim(-45, 280)
    ax.set_ylim(-75, 70)
    ax.text(
        0.5,
        -0.06,
        "947 个击球片段 · 30 fps\nRTMPose-M · COCO-17 关键点",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
        color=INK2,
        linespacing=1.4,
    )
    # panel 2
    ax = axs[1]
    ax.grid(False)
    N = c["N"]
    for p in (1, 2, 3):
        idx = np.where(ph == p)[0]
        ax.add_patch(
            Rectangle(
                (idx[0] - 0.5, 3.1),
                len(idx),
                0.6,
                fc=BLUE if p == 2 else "#9cbde9",
                ec="white",
                lw=1,
            )
        )
        if p == 2:
            ax.text(
                idx.mean(), 3.95, "前挥/触球", ha="center", va="center", fontsize=6.5, color=BLUE
            )
        else:
            ax.text(
                idx.mean(),
                3.4,
                {1: "引拍", 3: "随挥"}[p],
                ha="center",
                va="center",
                fontsize=6.5,
                color=INK,
            )
    for r, jn in enumerate(["shoulder", "elbow", "wrist"]):
        v = c["vis"][f"{arm}_{jn}"]
        y = 2.2 - r * 0.8
        for f in range(N):
            ax.add_patch(
                Rectangle(
                    (f - 0.5, y),
                    1,
                    0.55,
                    fc=ORANGE if v[f] != "visible" else "#e6e5e0",
                    ec="white",
                    lw=0.3,
                )
            )
        ax.text(
            -1.5,
            y + 0.27,
            {"shoulder": "肩", "elbow": "肘", "wrist": "腕"}[jn],
            ha="right",
            va="center",
            fontsize=7,
        )
    ax.set_xlim(-4, N)
    ax.set_ylim(0.5, 4.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, N - 1], ["0", f"{N - 1} 帧"])
    ax.tick_params(labelsize=6.5)
    for xx, colr, lb in ((0.25, ORANGE, "遮挡"), (0.55, "#d4d3cd", "可见")):
        ax.add_patch(
            Rectangle(
                (xx, -0.25), 0.05, 0.06, transform=ax.transAxes, fc=colr, ec="none", clip_on=False
            )
        )
        ax.text(xx + 0.07, -0.22, lb, transform=ax.transAxes, fontsize=6.6, va="center", color=INK2)
    # panel 3
    ax = axs[2]
    c2 = C["clip_000162"]
    ph2 = phases(c2)
    arm2 = "left" if c2["final"]["hitting_arm"] == "left" else "right"
    i2 = IDX[arm2]
    R = pd.read_pickle(work("e1_S.pkl"))
    r = R[
        (R.sid == "clip_000162")
        & (R.meth == "P")
        & (R.place == "random")
        & (R.L == 6)
        & (R.jset == "W")
    ].iloc[0]
    s, e = int(r.start), int(r.start) + 5
    tt = (c2["t"] - c2["t"][s]) * 1000
    ref = elbow_angle(c2["xy"][:, i2[0]], c2["xy"][:, i2[1]], c2["xy"][:, i2[2]])
    lo, hi = max(s - 3, 0), min(e + 3, c2["N"])
    ax.axvspan(tt[s] - 16, tt[e] + 16, color="#f3f2ee", lw=0)
    ax.text((tt[s] + tt[e]) / 2, 186, "删除腕坐标", ha="center", fontsize=6.3, color=INK2)
    ax.plot(tt[lo:hi], ref[lo:hi], color=INK, lw=1.4)
    k = i2[2]
    p_ = c2["xy"][:, k].copy()
    w_ = np.clip(np.nan_to_num(c2["sc"][:, k]), 0, 1)
    w_[w_ < 0.3] = 0
    w_[s : e + 1] = 0
    p_[s : e + 1] = np.nan
    for meth in ["linear", "U", "P"]:
        a = np.ones(c2["N"])
        a[ph2 == 1] = A1
        a[ph2 == 3] = A3
        rr = (
            interp_restore(c2["t"], p_, w_, "linear")
            if meth == "linear"
            else smooth_restore(c2["t"], p_, w_, LAM, a if meth == "P" else np.ones(c2["N"]))
        )
        est = elbow_angle(c2["xy"][s : e + 1, i2[0]], c2["xy"][s : e + 1, i2[1]], rr[s : e + 1])
        ax.plot(tt[s : e + 1], est, color=COL[meth], lw=1.5, marker="o", ms=2.5)
    ax.set_ylim(100, 192)
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("时间（ms）", fontsize=6.8, labelpad=1)
    ax.text(0.03, 0.97, "肘角（°）", transform=ax.transAxes, fontsize=6.5, color=INK2, va="top")
    hd = [
        Line2D([], [], color=INK, lw=1.4),
        Line2D([], [], color=MUTED, lw=1.5),
        Line2D([], [], color=ORANGE, lw=1.5),
        Line2D([], [], color=BLUE, lw=1.5),
    ]
    ax.legend(
        hd,
        ["删除前", "线性", "U", "P"],
        frameon=False,
        fontsize=6,
        loc="lower right",
        ncol=2,
        handlelength=1.2,
        columnspacing=0.8,
    )
    # panel 4
    ax = axs[3]
    vals = {"linear": 13.32, "pchip": 11.45, "U": 10.46, "P": 10.16}
    for j, (m, v) in enumerate(vals.items()):
        ax.bar(j, v, 0.66, color=COL[m])
        ax.text(j, v + 0.3, f"{v:.2f}", ha="center", fontsize=6.3, color=INK2)
    ax.set_xticks(range(4), ["线性", "PCHIP", "U", "P"], fontsize=6.8)
    ax.set_ylim(0, 16)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.annotate(
        "Δ = −0.30°",
        xy=(3, 10.9),
        xytext=(2.6, 14.3),
        fontsize=6.8,
        color=BLUE,
        weight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8),
    )
    for k in range(3):
        x0 = pos[k][0] + W + 0.004
        x1 = pos[k + 1][0] - 0.004
        fig.patches.append(
            FancyArrowPatch(
                (x0 + 0.004, T + H + 0.07),
                (x1 - 0.004, T + H + 0.07),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=10,
                color=MUTED,
                lw=1.1,
            )
        )
    fig.text(
        0.5,
        0.47,
        "阶段边界只输入 P；被删除的坐标只在计算误差时使用；参数按视频来源留一交叉验证选择",
        ha="center",
        fontsize=6.8,
        color=MUTED,
    )
    fig.text(L0, 0.39, "补充分析", fontsize=8.5, weight="bold")
    items = [
        ("阶段边界敏感性", "标注者 A / B 的边界、整体 ±2 帧、\n只用 f 和 c、阶段倍率错配"),
        ("跨模型参考", "RTMPose-x（384×288），\n同一视频、同一目标框"),
        ("真实遮挡片段", "仲裁判为遮挡的帧视为缺失，\n评价连续性与肢段长度"),
        ("四类击球识别", "击球臂特征 + 随机森林，\n按来源留一交叉验证"),
    ]
    for k, (h, b) in enumerate(items):
        x = L0 + k * (W + G)
        axb = fig.add_axes([x, 0.12, W + 0.02, 0.24])
        axb.axis("off")
        axb.add_patch(
            FancyBboxPatch(
                (0.02, 0.04),
                0.96,
                0.9,
                boxstyle="round,pad=0,rounding_size=0.06",
                fc="#f7f6f3",
                ec=LINE,
                lw=0.8,
                transform=axb.transAxes,
            )
        )
        axb.text(0.5, 0.72, h, ha="center", fontsize=7.6, weight="bold", transform=axb.transAxes)
        axb.text(
            0.5,
            0.36,
            b,
            ha="center",
            va="center",
            fontsize=6.4,
            color=INK2,
            linespacing=1.45,
            transform=axb.transAxes,
        )
    fig.savefig(OUT / "图1_实验流程图.png", bbox_inches="tight")
    plt.close()


# ======================= 图2 阶段与标注示例 =======================
def fig2():
    c = C[X1]
    ph = phases(c)
    arm = c["final"]["hitting_arm"]
    i = IDX[arm]
    N = c["N"]
    b = c["final"]["bounds"]
    B = {
        k: mid(b, k)
        for k in ["backswing_start", "forward_start", "contact_frame", "followthrough_end"]
    }
    fig = plt.figure(figsize=(7.3, 3.7))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.35, 1], hspace=0.15, left=0.07, right=0.98, top=0.9, bottom=0.12
    )
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    keys = [
        ("b 引拍起点", B["backswing_start"]),
        ("引拍中", (B["backswing_start"] + B["forward_start"]) // 2),
        ("f 前挥起点", B["forward_start"]),
        ("c 触球", B["contact_frame"]),
        ("随挥中", (B["contact_frame"] + B["followthrough_end"]) // 2),
        ("e 随挥结束", B["followthrough_end"]),
    ]

    def occmap(fi):
        joints = ("shoulder", "elbow", "wrist")
        return {k: c["vis"][f"{arm}_{j}"][fi] != "visible" for k, j in zip(i, joints, strict=True)}

    for k, (nm, fi) in enumerate(keys):
        skel(ax, c, fi, k * 95, arm, 0.8, occ=occmap(fi))
        ax.text(k * 95, -108, f"{nm}\n第 {fi} 帧", ha="center", va="top", fontsize=6.8, color=INK2)
    ax.set_aspect("equal")
    ax.set_ylim(-140, 80)
    ax.set_title(
        "(a) 关键帧骨架（近端球员，背对镜头；蓝色为击球臂，空心与虚线表示该帧被标为遮挡）",
        fontsize=8.5,
    )
    ax = fig.add_subplot(gs[1])
    ax.grid(False)
    for p, (nm, _col) in PH.items():
        idx = np.where(ph == p)[0]
        ax.add_patch(
            Rectangle(
                (idx[0] - 0.5, 3.0),
                len(idx),
                0.8,
                fc=BLUE if p == 2 else "#9cbde9",
                ec="white",
                lw=1.2,
            )
        )
        ax.text(
            idx.mean(),
            3.4,
            f"{nm}（{len(idx)} 帧）",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if p == 2 else INK,
        )
    for r, jn in enumerate(["shoulder", "elbow", "wrist"]):
        y = 2.1 - r * 0.8
        for _who, dy, vis in (("A", 0.36, c["visA"]), ("B", 0.0, c["visB"])):
            v = vis[f"{arm}_{jn}"]
            for f in range(N):
                ax.add_patch(
                    Rectangle(
                        (f - 0.5, y + dy),
                        1,
                        0.32,
                        fc=ORANGE if v[f] != "visible" else "#e6e5e0",
                        ec="white",
                        lw=0.4,
                    )
                )
        ax.text(
            -1,
            y + 0.34,
            {"shoulder": "肩", "elbow": "肘", "wrist": "腕"}[jn] + "  A/B",
            ha="right",
            va="center",
            fontsize=7,
        )
    for nm, fi in keys:
        if nm[0] in "bfce":
            ax.axvline(
                fi - 0.5 if nm[0] in "bf" else fi + 0.5,
                ymin=0.04,
                ymax=0.96,
                color=INK,
                lw=0.7,
                ls=(0, (2, 2)),
            )
    ax.set_xlim(-6, N)
    ax.set_ylim(0.3, 4.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("帧（30 fps）")
    ax.set_title("(b) 仲裁后的阶段划分与两名标注者的逐帧可见性（橙色为遮挡）", fontsize=8.5)
    fig.savefig(OUT / "图2_阶段划分与标注示例.png", bbox_inches="tight")
    plt.close()


# ======================= 图3 真实遮挡分布（重绘） =======================
def fig3():
    BINS = 10
    rows = []
    for sid, c in C.items():
        if c["group"] == "development_pilot_48" or not c["final"]["stroke_ok"]:
            continue
        ph = phases(c)
        if ph is None:
            continue
        for arm in arms(c):
            tau = np.full(c["N"], np.nan)
            for p in (1, 2, 3):
                idx = np.where(ph == p)[0]
                if len(idx):
                    tau[idx] = (p - 1) + np.arange(len(idx)) / len(idx)
            for jn in ("shoulder", "elbow", "wrist"):
                occ = np.isin(c["vis"][f"{arm}_{jn}"], ["occluded_self", "occluded_other"]).astype(
                    float
                )
                ok = np.isfinite(tau)
                bins = np.floor(tau[ok] * BINS).astype(int)
                for bb in np.unique(bins):
                    rows.append(
                        dict(
                            sid=sid,
                            stroke=c["stroke"],
                            joint=jn,
                            bin=bb,
                            occ=occ[ok][bins == bb].mean(),
                        )
                    )
    D = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 2, figsize=(7.3, 2.8), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axs[0]
    for p in (1, 2, 3):
        ax.axvspan(p - 1, p, color=PH[p][1], lw=0)
        ax.text(p - 0.5, 97, PH[p][0], ha="center", fontsize=7.5, color=INK2)
    jc = {"shoulder": AQUA, "elbow": ORANGE, "wrist": BLUE}
    jl = {"shoulder": "肩", "elbow": "肘", "wrist": "腕"}
    for jn in ("wrist", "elbow", "shoulder"):
        g = D[D.joint == jn].groupby("bin").occ
        m = g.mean() * 100
        se = g.sem() * 100 * 1.96
        x = (m.index + 0.5) / BINS
        ax.fill_between(x, m - se, m + se, color=jc[jn], alpha=0.15, lw=0)
        ax.plot(x, m, color=jc[jn], lw=2)
        ax.text(
            x[-1] + 0.04, m.iloc[-1], jl[jn], color=jc[jn], fontsize=8, va="center", weight="bold"
        )
    ax.set_xlim(0, 3.25)
    ax.set_ylim(0, 104)
    ax.set_xticks([0, 1, 2, 3], ["b", "f", "c", "e"])
    ax.set_ylabel("遮挡帧比例（%）")
    ax.set_xlabel("阶段内归一化时间")
    ax.set_title("(a) 击球臂关节遮挡随动作进程的变化")
    ax = axs[1]
    E0 = pd.read_csv(work("e0_occ.csv"))
    E0 = E0[E0.group != "development_pilot_48"]
    S = E0[E0.joint != "shoulder"].groupby(["stroke", "phase"]).occ.mean().unstack() * 100
    order = ["dink", "drop_shot", "forehand_drive", "volley"]
    S = S.loc[order]
    ax.imshow(
        S.values,
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f3f7fd", BLUE, "#123f7a"]),
        vmin=0,
        vmax=80,
        aspect="auto",
    )
    for r in range(4):
        for k in range(3):
            ax.text(
                k,
                r,
                f"{S.values[r, k]:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if S.values[r, k] > 45 else INK,
            )
    ax.set_xticks(range(3), ["引拍", "前挥/触球", "随挥"])
    ax.set_yticks(range(4), ["Dink", "Drop shot", "Forehand drive", "Volley"])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("(b) 肘与腕遮挡帧比例（%）")
    fig.tight_layout()
    fig.savefig(OUT / "图3_真实遮挡分布.png", bbox_inches="tight")
    plt.close()


# ======================= 图5 配对差分布 =======================
def fig5():
    S = pd.read_pickle(work("e1_summary.pkl"))
    d = S["d"]
    R = pd.read_pickle(work("e1_S.pkl"))
    keep = ((R.meth == "U") & np.isclose(R.lam, LAM)) | (
        (R.meth == "P") & np.isclose(R.lam, LAM) & (R.a1 == 2) & (R.a3 == 0.5)
    )
    X = R[keep & ~((R.place == "contact") & (R.L == 9))]
    fig, axs = plt.subplots(1, 2, figsize=(7.3, 2.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    rng = np.random.default_rng(1)
    for k, o in enumerate(["origin_01", "origin_02", "origin_03"]):
        v = d.xs(o, level="origin").values
        v = np.clip(v, -4, 4)
        ax.scatter(k + rng.uniform(-0.22, 0.22, len(v)), v, s=4, color=BLUE, alpha=0.25, lw=0)
        q = np.percentile(v, [25, 50, 75])
        ax.plot([k - 0.3, k + 0.3], [q[1]] * 2, color=INK, lw=1.4)
        ax.add_patch(Rectangle((k - 0.3, q[0]), 0.6, q[2] - q[0], fc="none", ec=INK, lw=0.9))
        ax.scatter(k, v.mean(), marker="D", s=22, color=ORANGE, zorder=5)
        ax.text(
            k,
            4.35,
            f"均值 {d.xs(o, level='origin').mean():+.2f}°",
            ha="center",
            fontsize=7,
            color=INK2,
        )
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(3), ["来源 1\n(n=210)", "来源 2\n(n=241)", "来源 3\n(n=154)"])
    ax.set_ylim(-4.3, 4.8)
    ax.set_ylabel("片段 Δ = MAE$_P$ − MAE$_U$（°）")
    ax.set_title("(a) 各来源片段配对差（负值表示 P 更好）")
    ax.scatter([], [], marker="D", color=ORANGE, label="均值")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax = axs[1]
    cs = X.groupby(["cp", "meth", "origin", "sid"]).mae.mean()
    for k, (cp, _nm) in enumerate([(1, "引拍"), (2, "前挥/触球"), (3, "随挥")]):
        dd = (
            cs.xs((cp, "P"), level=("cp", "meth")) - cs.xs((cp, "U"), level=("cp", "meth"))
        ).dropna()
        eq = dd.groupby("origin").mean().mean()
        bs = []
        rg = np.random.default_rng(20260907)
        gr = {
            o: dd.xs(o, level="origin").values for o in dd.index.get_level_values("origin").unique()
        }
        for _ in range(2000):
            bs.append(np.mean([rg.choice(v, len(v)).mean() for v in gr.values()]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.plot([lo, hi], [k, k], color=BLUE if cp == 2 else MUTED, lw=2.2, solid_capstyle="round")
        ax.scatter(eq, k, s=36, color=BLUE if cp == 2 else MUTED, zorder=5)
        ax.text(
            hi + 0.04, k, f"{eq:+.2f}°  [{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=7, color=INK2
        )
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(3), ["引拍", "前挥/触球", "随挥"])
    ax.invert_yaxis()
    ax.set_xlim(-1.0, 0.9)
    ax.set_xlabel("Δ = P − U（°），95% CI")
    ax.set_title("(b) 按删除窗口中心所在阶段")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "图5_配对差分布.png", bbox_inches="tight")
    plt.close()


# ======================= 图6 敏感性 =======================
def fig6():
    fig, axs = plt.subplots(1, 2, figsize=(7.3, 2.7), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    rows = [
        ("仲裁边界（主分析）", -0.297),
        ("标注者 A 边界", -0.224),
        ("标注者 B 边界", -0.314),
        ("只用 f 和 c", -0.237),
        ("整体提前 2 帧", -0.342),
        ("整体推后 2 帧", -0.033),
        ("阶段倍率错配", 0.291),
    ]
    for k, (_nm, v) in enumerate(rows):
        col = BLUE if v < 0 else ORANGE
        ax.plot([0, v], [k, k], color=col, lw=2.2, solid_capstyle="round")
        ax.scatter(v, k, s=30, color=col, zorder=5)
        ax.text(
            v + (0.025 if v >= 0 else -0.025),
            k,
            f"{v:+.2f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=7,
            color=INK2,
        )
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 0.45)
    ax.set_xlabel("Δ = P − U（°）")
    ax.set_title("(a) 阶段边界敏感性")
    ax.grid(axis="y", visible=False)
    ax = axs[1]
    thr = [0.3, 0.5, 0.7]
    allv = [-0.297, -0.144, 0.026]
    p2 = [-0.651, -0.393, -0.115]
    n = [605, 579, 337]
    ax.plot(thr, allv, color=MUTED, marker="o", lw=1.6, label="全部窗口")
    ax.plot(thr, p2, color=BLUE, marker="o", lw=1.8, label="前挥/触球阶段")
    for x, y, nn in zip(thr, allv, n, strict=True):
        ax.text(x, y + 0.05, f"n={nn}", ha="center", fontsize=6.5, color=INK2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(thr, ["≥0.3", "≥0.5", "≥0.7"])
    ax.set_xlabel("参考置信度阈值")
    ax.set_ylabel("Δ = P − U（°）")
    ax.set_ylim(-0.75, 0.2)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("(b) 参考置信度的影响")
    fig.tight_layout()
    fig.savefig(OUT / "图6_敏感性分析.png", bbox_inches="tight")
    plt.close()


def _keep(R):
    return (
        (R.meth.isin(["linear", "pchip"]))
        | ((R.meth == "U") & np.isclose(R.lam, SEL_U["lambda"]))
        | ((R.meth == "P") & np.isclose(R.lam, LAM) & (R.a1 == A1) & (R.a3 == A3))
    )


def fig4():
    R = pd.read_pickle(work("e1_S.pkl"))
    X = R[_keep(R)]

    def agg(df, by):
        return (
            df.groupby(by + ["meth", "origin", "sid"])
            .mae.mean()
            .groupby(by + ["meth", "origin"])
            .mean()
            .groupby(by + ["meth"])
            .mean()
            .unstack("meth")
        )

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    A = agg(X[(X.cp > 0) & ~((X.place == "contact") & (X.L == 9))], ["cp"])
    ax = axs[0]
    x = np.arange(3)
    w = 0.2
    for k, m in enumerate(["linear", "pchip", "U", "P"]):
        v = A.loc[[1, 2, 3], m].values
        ax.bar(x + (k - 1.5) * w, v, w - 0.03, color=COL[m], label=LAB[m])
        for xi, vi in zip(x + (k - 1.5) * w, v, strict=True):
            ax.text(
                xi,
                vi + 0.3,
                f"{vi:.1f}",
                ha="center",
                fontsize=5.8,
                color="#52514e",
                rotation=90,
                va="bottom",
            )
    ax.set_xticks(x, ["引拍", "前挥/触球", "随挥"])
    ax.set_ylabel("二维肘角 MAE（°）")
    ax.set_ylim(0, 22)
    ax.set_xlabel("掩码中心所在阶段")
    ax.set_title("(a) 按阶段", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    B = agg(X[X.place == "random"], ["L"])
    ax = axs[1]
    for m in ["linear", "pchip", "U", "P"]:
        ax.plot(
            B.index * 1000 / 30,
            B[m],
            marker="o",
            ms=4,
            lw=1.6 if m in "UP" else 1.2,
            color=COL[m],
            label=LAB[m],
        )
    ax.set_xlabel("掩码长度（ms）")
    ax.set_ylabel("二维肘角 MAE（°）")
    ax.set_xticks([67, 100, 200], ["67", "100", "200"])
    ax.set_xlim(50, 215)
    ax.set_title("(b) 按掩码长度（随机位置）", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "图4_合成遮挡误差.png", bbox_inches="tight")
    plt.close()


def fig7():
    R = pd.read_pickle(work("e1_S.pkl"))
    X = R[_keep(R)]
    P = X[
        (X.meth == "P") & (X.place == "random") & (X.L == 6) & (X.cp == 2) & (X.jset == "W")
    ].copy()
    U = X[(X.meth == "U") & (X.place == "random") & (X.L == 6) & (X.cp == 2) & (X.jset == "W")]
    P = P.merge(U[["sid", "arm", "start", "mae"]], on=["sid", "arm", "start"], suffixes=("", "_U"))
    P["d"] = P.mae - P.mae_U
    med = P.d.median()
    typ = P.iloc[(P.d - med).abs().argsort().iloc[0]]
    worst = P.sort_values("d").iloc[-1]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=False)
    for ax, (row, title) in zip(
        axs, [(typ, "(a) 典型例子（P−U 接近中位数）"), (worst, "(b) P 表现最差的例子")], strict=True
    ):
        c = C[row.sid]
        ph = phases(c)
        i = IDX[row.arm]
        t = (c["t"] - c["t"][0]) * 1000
        ref = elbow_angle(c["xy"][:, i[0]], c["xy"][:, i[1]], c["xy"][:, i[2]])
        for p, col in ((1, "#f3f2ee"), (2, "#e3ecf8"), (3, "#f3f2ee")):
            idx = np.where(ph == p)[0]
            if len(idx):
                ax.axvspan(t[idx[0]] - 16, t[idx[-1]] + 16, color=col, lw=0)
        s, e = int(row.start), int(row.start) + 5
        ax.axvspan(t[s] - 16, t[e] + 16, fc="none", ec="#c3c2b7", hatch="//", lw=0.0)
        ax.plot(t, ref, color="#0b0b0b", lw=1.4, label="参考（原始估计）")
        for meth in ["linear", "U", "P"]:
            k = i[2]
            p_ = c["xy"][:, k].copy()
            w_ = np.clip(np.nan_to_num(c["sc"][:, k]), 0, 1)
            w_[w_ < 0.3] = 0
            w_[s : e + 1] = 0
            p_[s : e + 1] = np.nan
            a = np.ones(c["N"])
            a[ph == 1] = A1
            a[ph == 3] = A3
            r = (
                interp_restore(c["t"], p_, w_, "linear")
                if meth == "linear"
                else smooth_restore(c["t"], p_, w_, LAM, a if meth == "P" else np.ones(c["N"]))
            )
            est = elbow_angle(c["xy"][s : e + 1, i[0]], c["xy"][s : e + 1, i[1]], r[s : e + 1])
            ax.plot(t[s : e + 1], est, color=COL[meth], lw=1.6, marker="o", ms=3, label=LAB[meth])
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("时间（ms）")
        ax.set_ylabel("二维肘角（°）")
        ax.text(
            0.02,
            0.04,
            f"{row.sid}  P {row.mae:.1f}° / U {row.mae_U:.1f}°",
            transform=ax.transAxes,
            fontsize=6.5,
            color="#52514e",
        )
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "图7_轨迹示例.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    for nm in sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7"]:
        globals()[nm]()
        print(nm, "ok")
