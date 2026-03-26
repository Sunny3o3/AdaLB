import os
import ast
import numpy as np
import matplotlib.pyplot as plt


# =========================================
# Config
# =========================================
RESULT_DIR = "./results_burgers_complete"
SAVE_DIR = "./figures_burgers_mean"

# 是否显示标准差阴影
SHOW_STD_BAND = False

# 是否使用对数纵轴
USE_LOG_SCALE = False

# 论文主比较图：建议保留这些方法
MAIN_COMPARE = [
    ("AdaLB", "adalb", 0.99),
    ("Adam", "adam", None),
    ("AdamW", "adamw", None),
    ("AdaBelief", "adabelief", None),
    ("NosAdam", "nosadam", None),
]

# gamma 敏感性图：只比较 AdaLB
GAMMA_COMPARE = [
    ("AdaLB ($\\gamma=0.8$)", "adalb", 0.8),
    ("AdaLB ($\\gamma=0.9$)", "adalb", 0.9),
    ("AdaLB ($\\gamma=0.99$)", "adalb", 0.99),
]

# hybrid / L-BFGS 图
HYBRID_COMPARE = [
    ("L-BFGS", "lbfgs", None),
    ("Adam→L-BFGS", "adam_lbfgs", None),
    ("AdaLB→L-BFGS ($\\gamma=0.99$)", "adalb_lbfgs", 0.99),
]

# 输出文件名
MAIN_FIG_NAME = "burgers_main_mean_curve.png"
MAIN_ZOOM_FIG_NAME = "burgers_main_mean_curve_zoom.png"
GAMMA_FIG_NAME = "burgers_gamma_mean_curve.png"
GAMMA_ZOOM_FIG_NAME = "burgers_gamma_mean_curve_zoom.png"
HYBRID_FIG_NAME = "burgers_hybrid_mean_curve.png"
HYBRID_ZOOM_FIG_NAME = "burgers_hybrid_mean_curve_zoom.png"


# =========================================
# Utils
# =========================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_loss_list(file_path: str) -> np.ndarray:
    """
    读取 loss_list.txt
    文件内容形如: [0.1, 0.08, 0.05, ...]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    data = ast.literal_eval(raw)
    return np.asarray(data, dtype=np.float64)


def format_gamma(gamma):
    """
    将 gamma 转成和目录名一致的字符串
    """
    if gamma is None:
        return None
    return str(gamma)


def collect_run_files(result_dir: str, optimizer: str, gamma=None):
    """
    从 results_burgers_complete 中收集匹配的 loss_list.txt 文件

    目录命名来自 train_new.py:
    {optimizer}_gamma{GAMMA}_seed{seed}
    """
    gamma_str = format_gamma(gamma)
    matched_files = []

    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    for name in os.listdir(result_dir):
        full_dir = os.path.join(result_dir, name)
        if not os.path.isdir(full_dir):
            continue

        if gamma_str is not None:
            prefix = f"{optimizer}_gamma{gamma_str}_seed"
        else:
            # gamma=None 时，兼容如 adam_gamma0.99_seed0 这种命名
            prefix = f"{optimizer}_"

        if name.startswith(prefix):
            loss_path = os.path.join(full_dir, "loss_list.txt")
            if os.path.isfile(loss_path):
                matched_files.append(loss_path)

    # 进一步过滤 gamma=None 的情况，避免误匹配
    if gamma is None:
        refined = []
        for fp in matched_files:
            base = os.path.basename(os.path.dirname(fp))
            if base.startswith(f"{optimizer}_gamma") and "_seed" in base:
                refined.append(fp)
        matched_files = refined

    matched_files.sort()
    return matched_files


def load_curves(result_dir: str, optimizer: str, gamma=None):
    """
    读取同一设置下的多个 seed 曲线，返回 shape = [num_runs, num_epochs]
    自动截断到最短长度，便于求 mean/std
    """
    files = collect_run_files(result_dir, optimizer, gamma)
    if len(files) == 0:
        raise FileNotFoundError(
            f"No loss files found for optimizer={optimizer}, gamma={gamma} in {result_dir}"
        )

    curves = [read_loss_list(fp) for fp in files]
    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]

    return np.vstack(curves), files


def compute_mean_std(curves: np.ndarray):
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)
    return mean_curve, std_curve


def plot_mean_curves(
    series_list,
    save_path: str,
    title: str,
    xlim=None,
    ylim=None,
    use_log_scale=True,
    show_std_band=True,
    dpi=300
):
    """
    series_list: [
        {
            "label": str,
            "mean": np.ndarray,
            "std": np.ndarray
        },
        ...
    ]
    """
    plt.figure(figsize=(10, 6))

    for item in series_list:
        x = np.arange(1, len(item["mean"]) + 1)
        y = item["mean"]
        s = item["std"]

        plt.plot(x, y, linewidth=1.5, label=item["label"])

        if show_std_band:
            lower = np.clip(y - s, a_min=0.0, a_max=None)
            upper = y + s
            plt.fill_between(x, lower, upper, alpha=0.15)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    if use_log_scale:
        plt.yscale("log")

    if xlim is not None:
        plt.xlim(*xlim)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def prepare_series(result_dir: str, config_list):
    """
    config_list: [
        (label, optimizer, gamma),
        ...
    ]
    """
    series = []
    for label, optimizer, gamma in config_list:
        curves, files = load_curves(result_dir, optimizer, gamma)
        mean_curve, std_curve = compute_mean_std(curves)

        print(f"[Loaded] {label}")
        print(f"  optimizer = {optimizer}, gamma = {gamma}")
        print(f"  runs = {len(files)}, epochs = {curves.shape[1]}")
        for fp in files:
            print(f"    - {fp}")

        series.append({
            "label": label,
            "mean": mean_curve,
            "std": std_curve
        })
    return series


# =========================================
# Main
# =========================================
if __name__ == "__main__":
    ensure_dir(SAVE_DIR)

    # -----------------------------
    # 1) Main comparison figure
    # -----------------------------
    main_series = prepare_series(RESULT_DIR, MAIN_COMPARE)

    plot_mean_curves(
        series_list=main_series,
        save_path=os.path.join(SAVE_DIR, MAIN_FIG_NAME),
        title="Burgers equation: mean loss curves over 5 seeds",
        xlim=None,
        ylim=None if USE_LOG_SCALE else (0, 0.3),
        use_log_scale=USE_LOG_SCALE,
        show_std_band=SHOW_STD_BAND
    )

    plot_mean_curves(
        series_list=main_series,
        save_path=os.path.join(SAVE_DIR, MAIN_ZOOM_FIG_NAME),
        title="Burgers equation: mean loss curves over 5 seeds (zoom)",
        xlim=(8000, 10000),
        ylim=(1e-4, 2e-2) if USE_LOG_SCALE else (0, 0.05),
        use_log_scale=USE_LOG_SCALE,
        show_std_band=SHOW_STD_BAND
    )

    # -----------------------------
    # 2) Gamma sensitivity figure
    # -----------------------------
    gamma_series = prepare_series(RESULT_DIR, GAMMA_COMPARE)

    plot_mean_curves(
        series_list=gamma_series,
        save_path=os.path.join(SAVE_DIR, GAMMA_FIG_NAME),
        title="Burgers equation: sensitivity to $\\gamma$ (mean over 5 seeds)",
        xlim=None,
        ylim=None if USE_LOG_SCALE else None,
        use_log_scale=USE_LOG_SCALE,
        show_std_band=SHOW_STD_BAND
    )

    plot_mean_curves(
        series_list=gamma_series,
        save_path=os.path.join(SAVE_DIR, GAMMA_ZOOM_FIG_NAME),
        title="Burgers equation: sensitivity to $\\gamma$ (zoom)",
        xlim=(8000, 10000),
        ylim=(1e-4, 2e-2) if USE_LOG_SCALE else None,
        use_log_scale=USE_LOG_SCALE,
        show_std_band=SHOW_STD_BAND
    )

    # -----------------------------
    # 3) Hybrid comparison
    # -----------------------------
    try:
        hybrid_series = prepare_series(RESULT_DIR, HYBRID_COMPARE)

        plot_mean_curves(
            series_list=hybrid_series,
            save_path=os.path.join(SAVE_DIR, HYBRID_FIG_NAME),
            title="Burgers equation: L-BFGS-based pipelines (mean over 5 seeds)",
            xlim=None,
            ylim=None if USE_LOG_SCALE else None,
            use_log_scale=USE_LOG_SCALE,
            show_std_band=SHOW_STD_BAND
        )

        plot_mean_curves(
            series_list=hybrid_series,
            save_path=os.path.join(SAVE_DIR, HYBRID_ZOOM_FIG_NAME),
            title="Burgers equation: L-BFGS-based pipelines (zoom)",
            xlim=(4000, 6000),
            ylim=(1e-5, 1e-2) if USE_LOG_SCALE else None,
            use_log_scale=USE_LOG_SCALE,
            show_std_band=SHOW_STD_BAND
        )
    except FileNotFoundError as e:
        print("\n[Warning] Hybrid figure skipped.")
        print(e)

    print("\nDone. Figures have been saved to:")
    print(os.path.abspath(SAVE_DIR))