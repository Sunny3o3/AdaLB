import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import math
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from optimal import AdaLB, AdaBelief, AdamW

try:
    from optimal import NosAdam
except ImportError:
    NosAdam = None


# =========================
# Config: edit here in PyCharm
# =========================
OPTIMIZER_NAME = "adalb_lbfgs"
# 可选:
# "adam", "amsgrad", "adamw", "adabelief", "nosadam",
# "adalb", "lbfgs", "adam_lbfgs", "adalb_lbfgs"

GAMMA = 0.9
EPOCHS = 10000
SEEDS = [0, 1, 2, 3, 4]
SAVE_CURVE = True
RESULT_DIR = "./results_high_order"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# problem setting
X_LEFT = 0.0
X_RIGHT = 1.0
N_INTERIOR = 50
N_TEST = 400
HIDDEN_UNITS = 20

# ---- L-BFGS settings ----
LBFGS_LR = 1.0
LBFGS_MAX_ITER = 20
LBFGS_HISTORY_SIZE = 100
LBFGS_LINE_SEARCH_FN = "strong_wolfe"

# ---- Auto stop for L-BFGS-like phase ----
ENABLE_LBFGS_AUTO_STOP = True
LBFGS_STOP_WINDOW = 50
LBFGS_STOP_REL_TOL = 1e-4
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


class ODE_Net(nn.Module):
    def __init__(self, hidden_units=20):
        super().__init__()
        self.layer1 = nn.Linear(1, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units)
        self.layer3 = nn.Linear(hidden_units, hidden_units)
        self.layer4 = nn.Linear(hidden_units, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        out = self.activation(self.layer1(x))
        out = self.activation(self.layer2(out))
        out = self.activation(self.layer3(out))
        out = self.layer4(out)
        return out


def exact_solution(x: torch.Tensor) -> torch.Tensor:
    # Analytical solution from your original code comment:
    # y(x) = 8 + 4x - 7e^x + 3xe^x
    return 8.0 + 4.0 * x - 7.0 * torch.exp(x) + 3.0 * x * torch.exp(x)


def residual(model, x: torch.Tensor) -> torch.Tensor:
    x = x.clone().detach().requires_grad_(True)
    y = model(x)

    y_x = torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]
    y_xx = torch.autograd.grad(
        y_x, x, grad_outputs=torch.ones_like(y_x), create_graph=True
    )[0]
    y_xxx = torch.autograd.grad(
        y_xx, x, grad_outputs=torch.ones_like(y_xx), create_graph=True
    )[0]
    y_xxxx = torch.autograd.grad(
        y_xxx, x, grad_outputs=torch.ones_like(y_xxx), create_graph=True
    )[0]

    # PDE: y_xxxx - 2 y_xxx + y_xx = 0
    res = y_xxxx - 2.0 * y_xxx + y_xx
    return res


def boundary_loss(model, device):
    x0 = torch.tensor([[0.0]], device=device, requires_grad=True)
    y0 = model(x0)

    y0_x = torch.autograd.grad(
        y0, x0, grad_outputs=torch.ones_like(y0), create_graph=True
    )[0]
    y0_xx = torch.autograd.grad(
        y0_x, x0, grad_outputs=torch.ones_like(y0_x), create_graph=True
    )[0]
    y0_xxx = torch.autograd.grad(
        y0_xx, x0, grad_outputs=torch.ones_like(y0_xx), create_graph=True
    )[0]

    # boundary conditions from original code
    bc1 = y0 - 1.0       # y(0) = 1
    bc2 = y0_x - 0.0     # y'(0) = 0
    bc3 = y0_xx - (-1.0) # y''(0) = -1
    bc4 = y0_xxx - 2.0   # y'''(0) = 2

    loss_bc = bc1.pow(2) + bc2.pow(2) + bc3.pow(2) + bc4.pow(2)
    return loss_bc.mean()


class PINNHighOrder:
    def __init__(self, optimizer_name="adalb", gamma=0.99):
        self.device = DEVICE
        self.optimizer_name = optimizer_name.lower()
        self.gamma = gamma

        self.model = ODE_Net(hidden_units=HIDDEN_UNITS).to(self.device)

        self.x_interior = torch.rand(N_INTERIOR, 1, device=self.device)
        self.loss_list = []
        self.loss_res_list = []
        self.loss_bc_list = []

        self.model_param_bytes = count_model_param_bytes(self.model)
        self.max_optimizer_state_bytes = 0

        self.trained_epochs = 0
        self.stopped_early = False
        self.early_stop_reason = ""

        self.optim = self.build_optimizer(self.optimizer_name)

    def build_optimizer(self, name: str):
        name = name.lower()

        if name == "adam":
            return optim.Adam(self.model.parameters(), lr=1e-3)

        elif name == "amsgrad":
            return optim.Adam(self.model.parameters(), lr=1e-3, amsgrad=True)

        elif name == "adamw":
            return AdamW(self.model.parameters())

        elif name == "adabelief":
            return AdaBelief(self.model.parameters())

        elif name == "nosadam":
            if NosAdam is None:
                raise ImportError("NosAdam is not available in your current optimal.py")
            return NosAdam(self.model.parameters())

        elif name == "adalb":
            return AdaLB(self.model.parameters(), lr=ADALB_STAGE_LR, gamma=self.gamma)

        elif name == "lbfgs":
            return optim.LBFGS(
                self.model.parameters(),
                lr=LBFGS_LR,
                max_iter=LBFGS_MAX_ITER,
                history_size=LBFGS_HISTORY_SIZE,
                line_search_fn=LBFGS_LINE_SEARCH_FN
            )

        elif name == "adam_lbfgs":
            return optim.Adam(self.model.parameters(), lr=ADAM_STAGE_LR)

        elif name == "adalb_lbfgs":
            return AdaLB(self.model.parameters(), lr=ADALB_STAGE_LR, gamma=self.gamma)

        else:
            raise ValueError(f"Unsupported optimizer: {name}")

    def compute_loss(self):
        r_interior = residual(self.model, self.x_interior)
        loss_res = torch.mean(r_interior ** 2)
        loss_bc = boundary_loss(self.model, self.device)
        loss = loss_res + loss_bc
        return loss, loss_res, loss_bc

    def update_optimizer_memory_record(self, optimizer):
        state_bytes = get_optimizer_state_bytes(optimizer)
        if state_bytes > self.max_optimizer_state_bytes:
            self.max_optimizer_state_bytes = state_bytes

    def train_one_epoch_standard(self, epoch_idx: int, total_epochs: int):
        self.optim.zero_grad()
        loss, loss_res, loss_bc = self.compute_loss()
        loss.backward()
        self.optim.step()

        self.loss_list.append(float(loss.item()))
        self.loss_res_list.append(float(loss_res.item()))
        self.loss_bc_list.append(float(loss_bc.item()))

        self.update_optimizer_memory_record(self.optim)

        if (epoch_idx + 1) % 500 == 0 or epoch_idx == 0:
            print(f"epoch {epoch_idx + 1}/{total_epochs}, loss = {loss.item():.6e}")

    def train_one_epoch_lbfgs(self, epoch_idx: int, total_epochs: int):
        last_loss = {"loss": None, "res": None, "bc": None}

        def closure():
            self.optim.zero_grad()
            loss, loss_res, loss_bc = self.compute_loss()
            loss.backward()

            last_loss["loss"] = float(loss.item())
            last_loss["res"] = float(loss_res.item())
            last_loss["bc"] = float(loss_bc.item())
            return loss

        self.optim.step(closure)

        self.loss_list.append(last_loss["loss"])
        self.loss_res_list.append(last_loss["res"])
        self.loss_bc_list.append(last_loss["bc"])

        self.update_optimizer_memory_record(self.optim)

        if (epoch_idx + 1) % 500 == 0 or epoch_idx == 0:
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

        if self.optimizer_name in ["adam", "amsgrad", "adamw", "adabelief", "nosadam", "adalb"]:
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
            self.optim = optim.Adam(self.model.parameters(), lr=ADAM_STAGE_LR)
            for epoch in range(adam_epochs):
                self.train_one_epoch_standard(epoch, epochs)
                self.trained_epochs += 1

            if lbfgs_epochs > 0:
                print(f"Phase 2: L-BFGS for {lbfgs_epochs} epochs")
                self.optim = optim.LBFGS(
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
                self.optim = optim.LBFGS(
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

        return time.time() - start_time

    def evaluate_model(self):
        self.model.eval()
        x_test = torch.linspace(X_LEFT, X_RIGHT, N_TEST, device=self.device).unsqueeze(1)

        with torch.no_grad():
            y_pred = self.model(x_test)
            y_true = exact_solution(x_test)

        y_pred_np = y_pred.detach().cpu().numpy().flatten()
        y_true_np = y_true.detach().cpu().numpy().flatten()

        test_mse = float(np.mean((y_pred_np - y_true_np) ** 2))
        rel_l2 = float(np.linalg.norm(y_pred_np - y_true_np) / (np.linalg.norm(y_true_np) + 1e-12))

        return x_test, y_pred_np, y_true_np, test_mse, rel_l2

    def get_memory_metrics(self):
        optimizer_state_bytes = self.max_optimizer_state_bytes
        return {
            "model_param_bytes": int(self.model_param_bytes),
            "model_param_mb": float(self.model_param_bytes / (1024 ** 2)),
            "optimizer_state_bytes": int(optimizer_state_bytes),
            "optimizer_state_mb": float(optimizer_state_bytes / (1024 ** 2)),
        }


def save_single_run(run_dir, pinn: PINNHighOrder, seed, train_time):
    os.makedirs(run_dir, exist_ok=True)

    x_test, y_pred_np, y_true_np, test_mse, rel_l2 = pinn.evaluate_model()
    memory_metrics = pinn.get_memory_metrics()
    actual_epochs = pinn.trained_epochs if pinn.trained_epochs > 0 else EPOCHS
    avg_time_per_epoch = train_time / actual_epochs if actual_epochs > 0 else 0.0

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
        "final_residual_loss": float(pinn.loss_res_list[-1]),
        "final_boundary_loss": float(pinn.loss_bc_list[-1]),
        "test_mse": float(test_mse),
        "rel_l2": float(rel_l2),

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

    with open(os.path.join(run_dir, "residual_loss_list.txt"), "w", encoding="utf-8") as f:
        f.write(str(pinn.loss_res_list))

    with open(os.path.join(run_dir, "boundary_loss_list.txt"), "w", encoding="utf-8") as f:
        f.write(str(pinn.loss_bc_list))

    if SAVE_CURVE:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(pinn.loss_list) + 1), pinn.loss_list, label="Total loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"Loss curve ({OPTIMIZER_NAME}, gamma={GAMMA}, seed={seed})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        x_test_np = x_test.detach().cpu().numpy().flatten()
        plt.plot(x_test_np, y_pred_np, label="PINN solution")
        plt.plot(x_test_np, y_true_np, "--", label="Analytical solution")
        plt.xlabel("x")
        plt.ylabel("y(x)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "solution_curve.png"), dpi=200)
        plt.close()

    return metrics


def summarize_metrics(all_metrics):
    def mean_std(key):
        arr = np.array([m[key] for m in all_metrics], dtype=float)
        return float(arr.mean()), float(arr.std())

    return {
        "optimizer": OPTIMIZER_NAME,
        "gamma": float(GAMMA),
        "configured_epochs": int(EPOCHS),
        "num_runs": len(all_metrics),
        "seeds": SEEDS,
        "device": str(DEVICE),

        "trained_epochs_mean": mean_std("trained_epochs")[0],
        "trained_epochs_std": mean_std("trained_epochs")[1],

        "final_loss_mean": mean_std("final_loss")[0],
        "final_loss_std": mean_std("final_loss")[1],

        "best_loss_mean": mean_std("best_loss")[0],
        "best_loss_std": mean_std("best_loss")[1],

        "final_residual_loss_mean": mean_std("final_residual_loss")[0],
        "final_residual_loss_std": mean_std("final_residual_loss")[1],

        "final_boundary_loss_mean": mean_std("final_boundary_loss")[0],
        "final_boundary_loss_std": mean_std("final_boundary_loss")[1],

        "test_mse_mean": mean_std("test_mse")[0],
        "test_mse_std": mean_std("test_mse")[1],

        "rel_l2_mean": mean_std("rel_l2")[0],
        "rel_l2_std": mean_std("rel_l2")[1],

        "train_time_mean": mean_std("train_time_sec")[0],
        "train_time_std": mean_std("train_time_sec")[1],

        "avg_time_per_epoch_mean": mean_std("avg_time_per_epoch_sec")[0],
        "avg_time_per_epoch_std": mean_std("avg_time_per_epoch_sec")[1],

        "optimizer_state_mb_mean": mean_std("optimizer_state_mb")[0],
        "optimizer_state_mb_std": mean_std("optimizer_state_mb")[1],

        "model_param_mb_mean": mean_std("model_param_mb")[0],
        "model_param_mb_std": mean_std("model_param_mb")[1],

        "all_runs": all_metrics,
    }


if __name__ == "__main__":
    os.makedirs(RESULT_DIR, exist_ok=True)

    all_metrics = []

    for i, seed in enumerate(SEEDS):
        print(f"\n[Run {i + 1}/{len(SEEDS)}] seed = {seed}")
        set_seed(seed)

        pinn = PINNHighOrder(optimizer_name=OPTIMIZER_NAME, gamma=GAMMA)
        train_time = pinn.train(epochs=EPOCHS)

        run_dir = os.path.join(
            RESULT_DIR,
            f"{OPTIMIZER_NAME}_gamma{GAMMA}_seed{seed}"
        )
        metrics = save_single_run(run_dir, pinn, seed, train_time)
        all_metrics.append(metrics)

    summary = summarize_metrics(all_metrics)

    summary_path = os.path.join(RESULT_DIR, f"summary_{OPTIMIZER_NAME}_gamma{GAMMA}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n================ Summary ================")
    print(f"optimizer = {OPTIMIZER_NAME}")
    print(f"gamma = {GAMMA}")
    print(f"configured_epochs = {EPOCHS}")
    print(f"trained_epochs    = {summary['trained_epochs_mean']:.1f} ± {summary['trained_epochs_std']:.1f}")
    print(f"seeds = {SEEDS}")
    print(f"final_loss   = {summary['final_loss_mean']:.6e} ± {summary['final_loss_std']:.6e}")
    print(f"best_loss    = {summary['best_loss_mean']:.6e} ± {summary['best_loss_std']:.6e}")
    print(f"residual     = {summary['final_residual_loss_mean']:.6e} ± {summary['final_residual_loss_std']:.6e}")
    print(f"boundary     = {summary['final_boundary_loss_mean']:.6e} ± {summary['final_boundary_loss_std']:.6e}")
    print(f"test_mse     = {summary['test_mse_mean']:.6e} ± {summary['test_mse_std']:.6e}")
    print(f"rel_l2       = {summary['rel_l2_mean']:.6e} ± {summary['rel_l2_std']:.6e}")
    print(f"time(sec)    = {summary['train_time_mean']:.4f} ± {summary['train_time_std']:.4f}")
    print(f"time/epoch   = {summary['avg_time_per_epoch_mean']:.6f} ± {summary['avg_time_per_epoch_std']:.6f}")
    print(f"opt_state_mb = {summary['optimizer_state_mb_mean']:.6f} ± {summary['optimizer_state_mb_std']:.6f}")
    print("=========================================\n")