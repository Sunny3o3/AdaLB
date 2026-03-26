import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

from network import Network
from NosAdam import NosAdam
from AdamW import AdamW
from AdaBelief import AdaBelief
from AdaLB_xr import AdaLB_xr
from AdaLB import AdaLB


# =========================
# Config: edit here in PyCharm
# =========================
OPTIMIZER_NAME = "adalb"
# 可选:
# "adam", "adamw", "adabelief", "nosadam",
# "adalb", "adalb_xr",
# "lbfgs", "adam_lbfgs", "adalb_lbfgs"

GAMMA = 0.9
EPOCHS = 10000
SEEDS = [0, 1, 2, 3, 4]
SAVE_CURVE = True
SAVE_MODEL = False

RESULT_DIR = "./results_burgers_complete"
REF_DATA_PATH = "./burgers_shock.mat"   # 参考解 .mat 文件路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- L-BFGS settings ----
LBFGS_LR = 1.0
LBFGS_MAX_ITER = 20
LBFGS_HISTORY_SIZE = 100
LBFGS_LINE_SEARCH_FN = "strong_wolfe"

# ---- Auto stop for L-BFGS-like phase ----
ENABLE_LBFGS_AUTO_STOP = False
LBFGS_STOP_WINDOW = 50
LBFGS_STOP_REL_TOL = 1e-5
LBFGS_STOP_MIN_EPOCHS = 50

# ---- Adam -> L-BFGS settings ----
ADAM_STAGE_EPOCHS = 5000
ADAM_STAGE_LR = 1e-3

# ---- AdaLB -> L-BFGS settings ----
ADALB_STAGE_EPOCHS = 5000
ADALB_STAGE_LR = 1e-3


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_optimizer_state_bytes(optimizer) -> int:
    state_bytes = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                state_bytes += value.numel() * value.element_size()
    return state_bytes


def count_model_param_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


class PINN:
    def __init__(self, optimizer_name="adalb", gamma=0.9):
        self.device = DEVICE
        self.optimizer_name = optimizer_name.lower()
        self.gamma = gamma

        self.model = Network(
            input_size=2,
            hidden_size=16,
            output_size=1,
            depth=8,
            act=torch.nn.Tanh
        ).to(self.device)

        # Burgers: x in [-1,1], t in [0,1], grid step 0.1
        self.h = 0.1
        self.k = 0.1
        x = torch.arange(-1, 1 + self.h, self.h)
        t = torch.arange(0, 1 + self.k, self.k)

        self.X_inside = torch.stack(
            torch.meshgrid(x, t, indexing='ij')
        ).reshape(2, -1).T

        bc1 = torch.stack(torch.meshgrid(x[0:1], t, indexing='ij')).reshape(2, -1).T
        bc2 = torch.stack(torch.meshgrid(x[-1:], t, indexing='ij')).reshape(2, -1).T
        ic = torch.stack(torch.meshgrid(x, t[0:1], indexing='ij')).reshape(2, -1).T

        self.X_boundary = torch.cat([bc1, bc2, ic], dim=0)

        u_bc1 = torch.zeros(len(bc1))
        u_bc2 = torch.zeros(len(bc2))
        u_ic = -torch.sin(math.pi * ic[:, 0])
        self.U_boundary = torch.cat([u_bc1, u_bc2, u_ic]).unsqueeze(1)

        self.X_inside = self.X_inside.to(self.device)
        self.X_boundary = self.X_boundary.to(self.device)
        self.U_boundary = self.U_boundary.to(self.device)
        self.X_inside.requires_grad_(True)

        self.criterion = torch.nn.MSELoss()

        self.loss_list = []
        self.loss_boundary_list = []
        self.loss_equation_list = []

        self.model_param_bytes = count_model_param_bytes(self.model)
        self.max_optimizer_state_bytes = 0

        self.trained_epochs = 0
        self.stopped_early = False
        self.early_stop_reason = ""

        self.optim = self.build_optimizer(self.optimizer_name)

    def build_optimizer(self, name: str):
        name = name.lower()

        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=1e-3)

        elif name == "adamw":
            return AdamW(self.model.parameters())

        elif name == "adabelief":
            return AdaBelief(self.model.parameters())

        elif name == "nosadam":
            return NosAdam(self.model.parameters())

        elif name == "adalb":
            return AdaLB(self.model.parameters(), gamma=self.gamma)

        elif name == "adalb_xr":
            return AdaLB_xr(self.model.parameters(), gamma=self.gamma, dynamic_beta2=False)

        elif name == "lbfgs":
            return torch.optim.LBFGS(
                self.model.parameters(),
                lr=LBFGS_LR,
                max_iter=LBFGS_MAX_ITER,
                history_size=LBFGS_HISTORY_SIZE,
                line_search_fn=LBFGS_LINE_SEARCH_FN
            )

        elif name == "adam_lbfgs":
            return torch.optim.Adam(self.model.parameters(), lr=ADAM_STAGE_LR)

        elif name == "adalb_lbfgs":
            return AdaLB(self.model.parameters(), lr=ADALB_STAGE_LR, gamma=self.gamma)

        else:
            raise ValueError(f"Unsupported optimizer: {name}")

    def compute_loss(self):
        # Boundary + Initial condition loss
        u_pred_boundary = self.model(self.X_boundary)
        loss_boundary = self.criterion(u_pred_boundary, self.U_boundary)

        # PDE residual loss
        u_inside = self.model(self.X_inside)

        du_dX = torch.autograd.grad(
            outputs=u_inside,
            inputs=self.X_inside,
            grad_outputs=torch.ones_like(u_inside),
            retain_graph=True,
            create_graph=True
        )[0]

        du_dx = du_dX[:, 0]
        du_dt = du_dX[:, 1]

        # Correct u_xx
        du_dxx = torch.autograd.grad(
            outputs=du_dx,
            inputs=self.X_inside,
            grad_outputs=torch.ones_like(du_dx),
            retain_graph=True,
            create_graph=True
        )[0][:, 0]

        # Burgers residual: u_t + u*u_x - nu*u_xx = 0, nu = 0.01/pi
        f = du_dt + u_inside.squeeze() * du_dx - (0.01 / math.pi) * du_dxx
        loss_equation = self.criterion(f, torch.zeros_like(f))

        loss = loss_equation + loss_boundary
        return loss, loss_boundary, loss_equation

    def update_optimizer_memory_record(self, optimizer):
        state_bytes = get_optimizer_state_bytes(optimizer)
        if state_bytes > self.max_optimizer_state_bytes:
            self.max_optimizer_state_bytes = state_bytes

    def train_one_epoch_standard(self, epoch_idx: int, total_epochs: int):
        self.optim.zero_grad()
        loss, loss_boundary, loss_equation = self.compute_loss()
        loss.backward()
        self.optim.step()

        self.loss_list.append(float(loss.item()))
        self.loss_boundary_list.append(float(loss_boundary.item()))
        self.loss_equation_list.append(float(loss_equation.item()))

        self.update_optimizer_memory_record(self.optim)

        if (epoch_idx + 1) % 100 == 0 or epoch_idx == 0:
            print(f"epoch {epoch_idx + 1}/{total_epochs}, loss = {loss.item():.6e}")

    def train_one_epoch_lbfgs(self, epoch_idx: int, total_epochs: int):
        last_loss = {"loss": None, "boundary": None, "equation": None}

        def closure():
            self.optim.zero_grad()
            loss, loss_boundary, loss_equation = self.compute_loss()
            loss.backward()

            last_loss["loss"] = float(loss.item())
            last_loss["boundary"] = float(loss_boundary.item())
            last_loss["equation"] = float(loss_equation.item())
            return loss

        self.optim.step(closure)

        self.loss_list.append(last_loss["loss"])
        self.loss_boundary_list.append(last_loss["boundary"])
        self.loss_equation_list.append(last_loss["equation"])

        self.update_optimizer_memory_record(self.optim)

        if (epoch_idx + 1) % 100 == 0 or epoch_idx == 0:
            print(f"epoch {epoch_idx + 1}/{total_epochs}, loss = {last_loss['loss']:.6e}")

    def should_stop_by_relative_window(self):
        if not ENABLE_LBFGS_AUTO_STOP:
            return False, ""

        if len(self.loss_list) < max(LBFGS_STOP_WINDOW, LBFGS_STOP_MIN_EPOCHS):
            return False, ""

        recent_losses = np.array(self.loss_list[-LBFGS_STOP_WINDOW:], dtype=np.float64)
        mean_loss = float(np.mean(recent_losses))
        loss_range = float(np.max(recent_losses) - np.min(recent_losses))
        rel_range = loss_range / max(abs(mean_loss), 1e-12)

        if rel_range < LBFGS_STOP_REL_TOL:
            reason = (
                f"lbfgs plateau: relative loss range over last {LBFGS_STOP_WINDOW} epochs "
                f"is {rel_range:.3e} < {LBFGS_STOP_REL_TOL:.3e}"
            )
            return True, reason

        return False, ""

    def train(self, epochs=10000):
        self.model.train()
        print("optimization")

        start_time = time.time()

        if self.optimizer_name in ["adam", "adamw", "adabelief", "nosadam", "adalb", "adalb_xr"]:
            for epoch in range(epochs):
                self.train_one_epoch_standard(epoch, epochs)
                self.trained_epochs += 1

        elif self.optimizer_name == "lbfgs":
            for epoch in range(epochs):
                self.train_one_epoch_lbfgs(epoch, epochs)
                self.trained_epochs += 1

                should_stop, reason = self.should_stop_by_relative_window()
                if should_stop:
                    self.stopped_early = True
                    self.early_stop_reason = reason
                    print(f"[Early Stop] epoch {epoch + 1}/{epochs}")
                    print(f"[Reason] {reason}")
                    break

        elif self.optimizer_name == "adam_lbfgs":
            adam_epochs = min(ADAM_STAGE_EPOCHS, epochs)
            lbfgs_epochs = max(0, epochs - adam_epochs)

            print(f"Phase 1: Adam for {adam_epochs} epochs")
            self.optim = torch.optim.Adam(self.model.parameters(), lr=ADAM_STAGE_LR)
            for epoch in range(adam_epochs):
                self.train_one_epoch_standard(epoch, epochs)
                self.trained_epochs += 1

            if lbfgs_epochs > 0:
                print(f"Phase 2: L-BFGS for {lbfgs_epochs} epochs")
                self.optim = torch.optim.LBFGS(
                    self.model.parameters(),
                    lr=LBFGS_LR,
                    max_iter=LBFGS_MAX_ITER,
                    history_size=LBFGS_HISTORY_SIZE,
                    line_search_fn=LBFGS_LINE_SEARCH_FN
                )

                for epoch in range(adam_epochs, epochs):
                    self.train_one_epoch_lbfgs(epoch, epochs)
                    self.trained_epochs += 1

                    should_stop, reason = self.should_stop_by_relative_window()
                    if should_stop:
                        self.stopped_early = True
                        self.early_stop_reason = "adam_lbfgs phase-2 " + reason
                        print(f"[Early Stop] epoch {epoch + 1}/{epochs}")
                        print(f"[Reason] {self.early_stop_reason}")
                        break

        elif self.optimizer_name == "adalb_lbfgs":
            adalb_epochs = min(ADALB_STAGE_EPOCHS, epochs)
            lbfgs_epochs = max(0, epochs - adalb_epochs)

            print(f"Phase 1: AdaLB for {adalb_epochs} epochs")
            self.optim = AdaLB(self.model.parameters(), lr=ADALB_STAGE_LR, gamma=self.gamma)
            for epoch in range(adalb_epochs):
                self.train_one_epoch_standard(epoch, epochs)
                self.trained_epochs += 1

            if lbfgs_epochs > 0:
                print(f"Phase 2: L-BFGS for {lbfgs_epochs} epochs")
                self.optim = torch.optim.LBFGS(
                    self.model.parameters(),
                    lr=LBFGS_LR,
                    max_iter=LBFGS_MAX_ITER,
                    history_size=LBFGS_HISTORY_SIZE,
                    line_search_fn=LBFGS_LINE_SEARCH_FN
                )

                for epoch in range(adalb_epochs, epochs):
                    self.train_one_epoch_lbfgs(epoch, epochs)
                    self.trained_epochs += 1

                    should_stop, reason = self.should_stop_by_relative_window()
                    if should_stop:
                        self.stopped_early = True
                        self.early_stop_reason = "adalb_lbfgs phase-2 " + reason
                        print(f"[Early Stop] epoch {epoch + 1}/{epochs}")
                        print(f"[Reason] {self.early_stop_reason}")
                        break

        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        end_time = time.time()
        return end_time - start_time

    def get_memory_metrics(self):
        optimizer_state_bytes = self.max_optimizer_state_bytes
        return {
            "model_param_bytes": int(self.model_param_bytes),
            "model_param_mb": float(self.model_param_bytes / (1024 ** 2)),
            "optimizer_state_bytes": int(optimizer_state_bytes),
            "optimizer_state_mb": float(optimizer_state_bytes / (1024 ** 2)),
        }

    def load_reference_solution(self, ref_path):
        """
        Expected .mat keys (common Burgers dataset):
            x    : shape (Nx, 1) or (Nx,)
            t    : shape (Nt, 1) or (Nt,)
            usol : shape (Nx, Nt)
        """
        data = loadmat(ref_path)

        if "x" not in data or "t" not in data or "usol" not in data:
            raise KeyError(
                f"Reference file {ref_path} must contain keys: 'x', 't', 'usol'. "
                f"Current keys: {list(data.keys())}"
            )

        x = data["x"].squeeze()
        t = data["t"].squeeze()
        usol = data["usol"]

        if usol.shape != (len(x), len(t)):
            raise ValueError(
                f"Expected usol shape = ({len(x)}, {len(t)}), but got {usol.shape}"
            )

        X, T = np.meshgrid(x, t, indexing="ij")
        X_test = np.stack([X.reshape(-1), T.reshape(-1)], axis=1)
        U_test = usol.reshape(-1, 1)

        X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        U_test = torch.tensor(U_test, dtype=torch.float32, device=self.device)

        return X_test, U_test

    def evaluate_relative_l2(self, ref_path):
        self.model.eval()
        X_test, U_test = self.load_reference_solution(ref_path)

        with torch.no_grad():
            U_pred = self.model(X_test)
            rel_l2 = torch.norm(U_pred - U_test, p=2) / torch.norm(U_test, p=2)

        self.model.train()
        return float(rel_l2.item())


def save_single_run(run_dir, pinn, seed, train_time):
    os.makedirs(run_dir, exist_ok=True)

    memory_metrics = pinn.get_memory_metrics()
    actual_epochs = pinn.trained_epochs if pinn.trained_epochs > 0 else EPOCHS
    avg_time_per_epoch = train_time / actual_epochs if actual_epochs > 0 else 0.0

    # Evaluate relative L2 error
    rel_l2_error = pinn.evaluate_relative_l2(REF_DATA_PATH)

    metrics = {
        "seed": int(seed),
        "optimizer": OPTIMIZER_NAME,
        "gamma": float(GAMMA),
        "configured_epochs": int(EPOCHS),
        "trained_epochs": int(actual_epochs),
        "stopped_early": bool(pinn.stopped_early),
        "early_stop_reason": pinn.early_stop_reason,
        "device": str(DEVICE),

        "final_loss": float(pinn.loss_list[-1]),
        "best_loss": float(min(pinn.loss_list)),
        "final_boundary_loss": float(pinn.loss_boundary_list[-1]),
        "final_equation_loss": float(pinn.loss_equation_list[-1]),
        "rel_l2_error": float(rel_l2_error),

        "train_time_sec": float(train_time),
        "avg_time_per_epoch_sec": float(avg_time_per_epoch),

        "model_param_bytes": memory_metrics["model_param_bytes"],
        "model_param_mb": memory_metrics["model_param_mb"],
        "optimizer_state_bytes": memory_metrics["optimizer_state_bytes"],
        "optimizer_state_mb": memory_metrics["optimizer_state_mb"],

        "lbfgs_lr": float(LBFGS_LR),
        "lbfgs_max_iter": int(LBFGS_MAX_ITER),
        "lbfgs_history_size": int(LBFGS_HISTORY_SIZE),
        "lbfgs_auto_stop_enabled": bool(ENABLE_LBFGS_AUTO_STOP),
        "lbfgs_stop_window": int(LBFGS_STOP_WINDOW),
        "lbfgs_stop_rel_tol": float(LBFGS_STOP_REL_TOL),
        "lbfgs_stop_min_epochs": int(LBFGS_STOP_MIN_EPOCHS),

        "adam_stage_epochs": int(ADAM_STAGE_EPOCHS) if OPTIMIZER_NAME == "adam_lbfgs" else 0,
        "adam_stage_lr": float(ADAM_STAGE_LR) if OPTIMIZER_NAME == "adam_lbfgs" else 0.0,

        "adalb_stage_epochs": int(ADALB_STAGE_EPOCHS) if OPTIMIZER_NAME == "adalb_lbfgs" else 0,
        "adalb_stage_lr": float(ADALB_STAGE_LR) if OPTIMIZER_NAME == "adalb_lbfgs" else 0.0,
    }

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(run_dir, "loss_list.txt"), "w", encoding="utf-8") as f:
        f.write(str(pinn.loss_list))

    with open(os.path.join(run_dir, "boundary_loss_list.txt"), "w", encoding="utf-8") as f:
        f.write(str(pinn.loss_boundary_list))

    with open(os.path.join(run_dir, "equation_loss_list.txt"), "w", encoding="utf-8") as f:
        f.write(str(pinn.loss_equation_list))

    if SAVE_MODEL:
        torch.save(pinn.model.state_dict(), os.path.join(run_dir, "model.pt"))

    if SAVE_CURVE:
        plt.figure()
        plt.title(f"Loss (seed={seed})")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(range(1, len(pinn.loss_list) + 1), pinn.loss_list)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=200)
        plt.close()

    return metrics


def save_summary(all_metrics):
    os.makedirs(RESULT_DIR, exist_ok=True)

    final_losses = np.array([m["final_loss"] for m in all_metrics], dtype=np.float64)
    best_losses = np.array([m["best_loss"] for m in all_metrics], dtype=np.float64)
    boundary_losses = np.array([m["final_boundary_loss"] for m in all_metrics], dtype=np.float64)
    equation_losses = np.array([m["final_equation_loss"] for m in all_metrics], dtype=np.float64)
    rel_l2_errors = np.array([m["rel_l2_error"] for m in all_metrics], dtype=np.float64)
    train_times = np.array([m["train_time_sec"] for m in all_metrics], dtype=np.float64)
    avg_epoch_times = np.array([m["avg_time_per_epoch_sec"] for m in all_metrics], dtype=np.float64)
    trained_epochs = np.array([m["trained_epochs"] for m in all_metrics], dtype=np.float64)
    optimizer_state_mbs = np.array([m["optimizer_state_mb"] for m in all_metrics], dtype=np.float64)
    model_param_mbs = np.array([m["model_param_mb"] for m in all_metrics], dtype=np.float64)

    summary = {
        "optimizer": OPTIMIZER_NAME,
        "gamma": float(GAMMA),
        "configured_epochs": int(EPOCHS),
        "num_runs": len(all_metrics),
        "seeds": SEEDS,
        "device": str(DEVICE),

        "trained_epochs_mean": float(trained_epochs.mean()),
        "trained_epochs_std": float(trained_epochs.std()),

        "final_loss_mean": float(final_losses.mean()),
        "final_loss_std": float(final_losses.std()),

        "best_loss_mean": float(best_losses.mean()),
        "best_loss_std": float(best_losses.std()),

        "boundary_loss_mean": float(boundary_losses.mean()),
        "boundary_loss_std": float(boundary_losses.std()),

        "equation_loss_mean": float(equation_losses.mean()),
        "equation_loss_std": float(equation_losses.std()),

        "rel_l2_error_mean": float(rel_l2_errors.mean()),
        "rel_l2_error_std": float(rel_l2_errors.std()),

        "train_time_mean": float(train_times.mean()),
        "train_time_std": float(train_times.std()),

        "avg_time_per_epoch_mean": float(avg_epoch_times.mean()),
        "avg_time_per_epoch_std": float(avg_epoch_times.std()),

        "optimizer_state_mb_mean": float(optimizer_state_mbs.mean()),
        "optimizer_state_mb_std": float(optimizer_state_mbs.std()),

        "model_param_mb_mean": float(model_param_mbs.mean()),
        "model_param_mb_std": float(model_param_mbs.std()),

        "all_runs": all_metrics
    }

    with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n================ Summary ================")
    print(f"optimizer = {OPTIMIZER_NAME}")
    print(f"gamma = {GAMMA}")
    print(f"configured_epochs = {EPOCHS}")
    print(f"trained_epochs    = {summary['trained_epochs_mean']:.1f} ± {summary['trained_epochs_std']:.1f}")
    print(f"seeds = {SEEDS}")
    print(f"final_loss         = {summary['final_loss_mean']:.6e} ± {summary['final_loss_std']:.6e}")
    print(f"best_loss          = {summary['best_loss_mean']:.6e} ± {summary['best_loss_std']:.6e}")
    print(f"boundary_loss      = {summary['boundary_loss_mean']:.6e} ± {summary['boundary_loss_std']:.6e}")
    print(f"equation_loss      = {summary['equation_loss_mean']:.6e} ± {summary['equation_loss_std']:.6e}")
    print(f"rel_l2_error       = {summary['rel_l2_error_mean']:.6e} ± {summary['rel_l2_error_std']:.6e}")
    print(f"train_time(sec)    = {summary['train_time_mean']:.4f} ± {summary['train_time_std']:.4f}")
    print(f"time/epoch(sec)    = {summary['avg_time_per_epoch_mean']:.6f} ± {summary['avg_time_per_epoch_std']:.6f}")
    print(f"optimizer_state_mb = {summary['optimizer_state_mb_mean']:.6f} ± {summary['optimizer_state_mb_std']:.6f}")
    print("=========================================\n")


if __name__ == "__main__":
    os.makedirs(RESULT_DIR, exist_ok=True)

    if not os.path.exists(REF_DATA_PATH):
        raise FileNotFoundError(
            f"Reference solution file not found: {REF_DATA_PATH}\n"
            f"Please place burgers_shock.mat at this path or modify REF_DATA_PATH."
        )

    all_metrics = []

    for i, seed in enumerate(SEEDS):
        print(f"\n[Run {i + 1}/{len(SEEDS)}] seed = {seed}")
        set_seed(seed)

        pinn = PINN(optimizer_name=OPTIMIZER_NAME, gamma=GAMMA)
        train_time = pinn.train(epochs=EPOCHS)

        run_dir = os.path.join(
            RESULT_DIR,
            f"{OPTIMIZER_NAME}_gamma{GAMMA}_seed{seed}"
        )

        metrics = save_single_run(run_dir, pinn, seed, train_time)
        all_metrics.append(metrics)

    save_summary(all_metrics)