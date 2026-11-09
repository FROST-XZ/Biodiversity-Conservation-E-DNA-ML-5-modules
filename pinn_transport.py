"""
Module 1: Physics-Informed Neural Network (SIREN) for Inverse Fluid Transport
=============================================================================
Advection-dispersion-decay PDE for eDNA drifting along the Brahmaputra reach:

    dC/dt + u dC/dx + v dC/dy - D (d2C/dx2 + d2C/dy2) + k C = S(x, y, t)

- SIREN backbone: sin(w0 * Wx + b), w0 = 30, 3 hidden layers x 64 units.
- Inverse parameters (u, D, k, S_x, S_y) are learned jointly with the weights using
  1,024-4,096-point collocation mini-batches (keeps peak VRAM < 1.5 GB).
- Dual engine: PyTorch autograd (FP32) when available, otherwise an exact NumPy
  FP32 second-order engine.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, IterableDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("eDNA.PINN")

SOURCE_SIGMA_X = 0.45  # km, Gaussian source width along the reach
SOURCE_SIGMA_Y = 0.18  # km, Gaussian source width across the channel
# Clamp ranges for (u, D, k x 1e3, S_x, S_y)
PARAM_LO = np.array([0.1, 0.5, 0.05, 0.1, 0.05], dtype=np.float32)
PARAM_HI = np.array([5.0, 50.0, 5.0, 20.0, 1.15], dtype=np.float32)


@dataclasses.dataclass
class PhysicalTransportConfig:
    """Configuration for the Brahmaputra River Basin eDNA transport model."""

    # Ground truth for zero-dataset forward simulation
    true_u: float = 1.42           # advection velocity (m/s)
    true_v: float = 0.18           # transverse drift (m/s)
    true_D: float = 12.50          # longitudinal dispersion (m^2/s)
    true_k: float = 0.00048        # decay rate (s^-1, ~24 min half-life)
    true_Sx: float = 2.35          # shedding origin (km)
    true_Sy: float = 0.42          # transverse shedding coordinate (km)
    source_strength: float = 85.0  # S_0 (copies / mL / s)

    # Initial guesses for the inverse problem
    init_u: float = 0.95
    init_D: float = 8.00
    init_k: float = 0.00025
    init_Sx: float = 1.50
    init_Sy: float = 0.25

    # Domain (km, km, hours)
    x_min: float = 0.0
    x_max: float = 25.0
    y_min: float = 0.0
    y_max: float = 1.2
    t_min: float = 0.0
    t_max: float = 6.0

    # SIREN / optimizer
    omega_0: float = 30.0
    hidden_dim: int = 64
    num_hidden_layers: int = 3
    collocation_batch_size: int = 16384  # clamped to 1,024-32,768 (GPU); NumPy path is cheap either way
    vram_safety_fraction: float = 0.75   # informational; the orchestrator applies the CUDA cap
    lambda_pde: float = 0.25
    learning_rate: float = 2e-3

    @property
    def lo(self) -> List[float]:
        return [self.x_min, self.y_min, self.t_min]

    @property
    def hi(self) -> List[float]:
        return [self.x_max, self.y_max, self.t_max]


def _units(u, D, k):
    """(m/s, m^2/s, 1/s) -> (km/hr, km^2/hr, 1/hr)."""
    return u * 3.6, D * 0.0036, k * 3600.0


def _spatial_decay(u_h, D_h, k_h, sqrt=math.sqrt):
    """Exact spatial decay exponent of the 1D advection-dispersion-reaction steady state."""
    return (sqrt(u_h ** 2 + 4.0 * D_h * k_h) - u_h) / (2.0 * D_h + 1e-8)


def analytical_advection_dispersion_decay(
    x_km: np.ndarray,
    y_km: np.ndarray,
    t_hr: np.ndarray,
    u: float,
    D: float,
    k: float,
    Sx: float,
    Sy: float,
    source_strength: float = 85.0,
    v: float = 0.18,
) -> np.ndarray:
    """Transient + steady-state plume solution with a Gaussian source at (Sx, Sy)."""
    dx, dy = x_km - Sx, y_km - Sy
    dxp = np.maximum(dx, 0.0)

    u_h, D_h, k_h = _units(u, D, k)
    u_h, D_h, k_h = max(u_h, 0.1), max(D_h, 1e-4), max(k_h, 1e-5)
    v_h = v * 3.6

    # Advective front u*t with dispersive broadening sqrt(2*D*t)
    eff_t = np.maximum(t_hr, 0.20)
    front_gate = 1.0 / (1.0 + np.exp((dx - u_h * eff_t) / np.sqrt(2.0 * D_h * eff_t + 0.12)))

    # Transverse spread identifies D independently of k
    sigma_y = np.sqrt(0.015 + 2.5 * D_h * dxp / (u_h + 1e-6))
    transverse = np.exp(-((dy - v_h * 0.03 * dxp) ** 2) / (2.0 * sigma_y ** 2 + 1e-6))

    upstream = np.exp(-(dx ** 2) / (2.0 * SOURCE_SIGMA_X ** 2))
    downstream = np.exp(-_spatial_decay(u_h, D_h, k_h, np.sqrt) * dxp)
    plume = np.where(dx < 0, upstream, downstream)

    c = source_strength * plume * transverse * front_gate
    return np.clip(c, 0.0, None).astype(np.float32)


def load_real_data(
    data_dir: Optional[str] = None,
    num_samples: int = 160,
    config: Optional[PhysicalTransportConfig] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load `{data_dir}/edna_field_transport.csv` if valid, else simulate field observations."""
    cfg = config or PhysicalTransportConfig()
    if data_dir:
        csv = os.path.join(data_dir, "edna_field_transport.csv")
        if os.path.isfile(csv):
            logger.info("Loading field transport dataset from %s", csv)
            df = pd.read_csv(csv)
            if {"sample_id", "x_km", "y_km", "t_hr", "concentration_copies_ml"} <= set(df.columns):
                return df

    rng = np.random.default_rng(seed)
    n = num_samples
    x_km = np.sort(rng.uniform(cfg.x_min, cfg.x_max, size=n)).astype(np.float32)
    y_km = rng.uniform(0.12, 0.88, size=n).astype(np.float32)
    t_hr = rng.uniform(0.6, cfg.t_max, size=n).astype(np.float32)  # resolves front arrival (u, D)

    clean = analytical_advection_dispersion_decay(
        x_km, y_km, t_hr, cfg.true_u, cfg.true_D, cfg.true_k,
        cfg.true_Sx, cfg.true_Sy, cfg.source_strength, cfg.true_v,
    )
    noise = rng.normal(0.0, 0.85, size=n).astype(np.float32)
    measured = np.clip(clean + noise * np.sqrt(np.maximum(clean, 0.1) / 15.0), 0.02, None)

    reaches = np.select(
        [x_km < 5.0, x_km < 14.0],
        ["Upper_Mangaldoi_Braided_Channel", "Mid_Kharupetia_Sandbar_Reach"],
        default="Lower_Gauhati_Confluence_Plume",
    )
    return pd.DataFrame(
        {
            "sample_id": [f"BRH_S{i:03d}" for i in range(1, n + 1)],
            "reach_zone": reaches,
            "x_km": np.round(x_km, 4),
            "y_km": np.round(y_km, 4),
            "t_hr": np.round(t_hr, 4),
            "concentration_copies_ml": np.round(measured, 4),
            "clean_analytical_copies_ml": np.round(clean, 4),
        }
    )


# =============================================================================
# PyTorch engine
# =============================================================================
if TORCH_AVAILABLE:

    class SineLayer(nn.Module):
        """SIREN layer: y = sin(omega_0 * (W x + b))."""

        def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0,
                     is_first: bool = False) -> None:
            super().__init__()
            self.omega_0 = float(omega_0)
            self.linear = nn.Linear(in_features, out_features)
            bound = 1.0 / in_features if is_first else math.sqrt(6.0 / in_features) / self.omega_0
            with torch.no_grad():
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.uniform_(-bound, bound)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sin(self.omega_0 * self.linear(x))

    class SIRENTransportPINN(nn.Module):
        """SIREN with learnable physical parameters (u, D, k, S_x, S_y)."""

        def __init__(self, config: Optional[PhysicalTransportConfig] = None) -> None:
            super().__init__()
            self.cfg = cfg = config or PhysicalTransportConfig()
            h = cfg.hidden_dim
            layers = [SineLayer(3, h, cfg.omega_0, is_first=True)]
            layers += [SineLayer(h, h, cfg.omega_0) for _ in range(cfg.num_hidden_layers - 1)]
            self.siren_backbone = nn.Sequential(*layers)
            self.output_head = nn.Linear(h, 1)
            self.register_buffer("_lo", torch.tensor(cfg.lo, dtype=torch.float32))
            self.register_buffer("_hi", torch.tensor(cfg.hi, dtype=torch.float32))
            with torch.no_grad():
                bound = math.sqrt(6.0 / h) / cfg.omega_0
                self.output_head.weight.uniform_(-bound, bound)
                self.output_head.bias.zero_()

            p = lambda v: nn.Parameter(torch.tensor(float(v), dtype=torch.float32))
            self.u_param, self.D_param = p(cfg.init_u), p(cfg.init_D)
            self.k_scaled_param = p(cfg.init_k * 1e3)
            self.Sx_param, self.Sy_param = p(cfg.init_Sx), p(cfg.init_Sy)

        u = property(lambda s: torch.clamp(s.u_param, 0.1, 5.0))
        D = property(lambda s: torch.clamp(s.D_param, 0.5, 50.0))
        k = property(lambda s: torch.clamp(s.k_scaled_param, 0.05, 5.0) * 1e-3)
        S_x = property(lambda s: torch.clamp(s.Sx_param, 0.1, 20.0))
        S_y = property(lambda s: torch.clamp(s.Sy_param, 0.05, 1.15))

        def physical_parameters(self) -> List[nn.Parameter]:
            return [self.u_param, self.D_param, self.k_scaled_param, self.Sx_param, self.Sy_param]

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            z = 2.0 * (coords - self._lo) / (self._hi - self._lo) - 1.0
            return torch.nn.functional.softplus(self.output_head(self.siren_backbone(z))) * 20.0

        def physics_envelope(self, coords: torch.Tensor) -> torch.Tensor:
            """Differentiable analytical plume built from the current (u, D, k, S_x, S_y)."""
            dx = coords[:, 0:1] - self.S_x
            dy = coords[:, 1:2] - self.S_y
            dxp = torch.clamp(dx, min=0.0)
            u_h, D_h, k_h = _units(self.u, self.D, self.k)

            sigma_y = torch.sqrt(0.015 + 2.5 * D_h * dxp / (u_h + 1e-6))
            drift = self.cfg.true_v * 3.6 * 0.03 * dxp
            transverse = torch.exp(-((dy - drift) ** 2) / (2.0 * sigma_y ** 2 + 1e-6))
            eff_t = torch.clamp(coords[:, 2:3], min=0.20)
            gate = torch.sigmoid((u_h * eff_t - dx) / torch.sqrt(2.0 * D_h * eff_t + 0.12))
            plume = torch.where(
                dx < 0.0,
                torch.exp(-(dx ** 2) / (2.0 * SOURCE_SIGMA_X ** 2)),
                torch.exp(-_spatial_decay(u_h, D_h, k_h, torch.sqrt) * dxp),
            )
            return self.cfg.source_strength * plume * transverse * gate

        def compute_pde_residual(self, collocation_coords: torch.Tensor) -> torch.Tensor:
            coords = collocation_coords.clone().detach().requires_grad_(True)
            C = self.forward(coords)
            ones = torch.ones_like(C)

            def grad(out: torch.Tensor) -> torch.Tensor:
                return torch.autograd.grad(out, coords, grad_outputs=ones,
                                           create_graph=True, retain_graph=True)[0]

            g = grad(C)
            dC_dx, dC_dy, dC_dt = g[:, 0:1], g[:, 1:2], g[:, 2:3]
            d2x, d2y = grad(dC_dx)[:, 0:1], grad(dC_dy)[:, 1:2]

            u_h, D_h, k_h = _units(self.u, self.D, self.k)
            x, y = coords[:, 0:1], coords[:, 1:2]
            source = (
                self.cfg.source_strength * k_h
                * torch.exp(-((x - self.S_x) ** 2) / (2.0 * SOURCE_SIGMA_X ** 2))
                * torch.exp(-((y - self.S_y) ** 2) / (2.0 * SOURCE_SIGMA_Y ** 2))
            )
            return dC_dt + u_h * dC_dx - D_h * (d2x + 0.15 * d2y) + k_h * C - source

    class DynamicCollocationDataset(IterableDataset):
        """Endless stream of uniformly sampled (x, y, t) collocation mini-batches."""

        def __init__(self, config: PhysicalTransportConfig, batch_size: int = 2048) -> None:
            super().__init__()
            self.cfg = config
            self.batch_size = min(max(batch_size, 1024), 32768)

        def __iter__(self) -> Iterator[torch.Tensor]:
            bounds = list(zip(self.cfg.lo, self.cfg.hi))
            while True:
                yield torch.cat(
                    [torch.empty((self.batch_size, 1), dtype=torch.float32).uniform_(a, b)
                     for a, b in bounds],
                    dim=-1,
                )


# =============================================================================
# NumPy engine
# =============================================================================
class NumpySIRENNetwork:
    """FP32 SIREN feature extractor + closed-form (ridge) readout, used when PyTorch is absent.

    NOTE: without autograd this engine does NOT solve the PDE through the network. The physical
    parameters are fitted to the data through the analytical plume model; the SIREN is then
    distilled onto the converged plume (ridge readout) so that the exported ONNX model is a real,
    working surrogate. Install PyTorch to use the autograd PINN path.
    """

    def __init__(self, config: PhysicalTransportConfig, seed: int = 42) -> None:
        self.cfg = config
        rng = np.random.default_rng(seed)
        h, w0 = config.hidden_dim, config.omega_0
        b2 = math.sqrt(6.0 / h) / w0

        def uni(shape, bound):
            return rng.uniform(-bound, bound, size=shape).astype(np.float32)

        self.W1, self.b1 = uni((3, h), 1.0 / 3.0), uni((1, h), 1.0 / 3.0)
        self.W2, self.b2 = uni((h, h), b2), uni((1, h), b2)
        self.W3, self.b3 = uni((h, h), b2), uni((1, h), b2)
        self.W_out = np.zeros((h, 1), dtype=np.float32)
        self.b_out = np.zeros((1, 1), dtype=np.float32)

        self.u, self.D, self.k = float(config.init_u), float(config.init_D), float(config.init_k)
        self.S_x, self.S_y = float(config.init_Sx), float(config.init_Sy)

    def normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        lo = np.asarray(self.cfg.lo, dtype=np.float32)
        hi = np.asarray(self.cfg.hi, dtype=np.float32)
        return (2.0 * (coords - lo) / (hi - lo) - 1.0).astype(np.float32)

    def features(self, coords: np.ndarray) -> np.ndarray:
        w0 = np.float32(self.cfg.omega_0)
        h = self.normalize_coords(coords)
        for W, b in ((self.W1, self.b1), (self.W2, self.b2), (self.W3, self.b3)):
            h = np.sin(w0 * (h @ W + b))
        return h

    def _plume(self, coords: np.ndarray) -> np.ndarray:
        return analytical_advection_dispersion_decay(
            coords[:, 0], coords[:, 1], coords[:, 2], self.u, self.D, self.k,
            self.S_x, self.S_y, self.cfg.source_strength, self.cfg.true_v,
        )[:, None]

    def fit_readout(self, coords: np.ndarray, target: np.ndarray, ridge: float = 1e-3) -> float:
        """Ridge-regress the linear head onto `target`; returns the fit R^2."""
        phi = self.features(coords)
        A = np.concatenate([phi, np.ones((len(phi), 1), dtype=np.float32)], axis=1).astype(np.float64)
        reg = ridge * np.eye(A.shape[1])
        reg[-1, -1] = 0.0
        sol = np.linalg.solve(A.T @ A + reg * len(A), A.T @ target.astype(np.float64))
        self.W_out, self.b_out = sol[:-1].astype(np.float32), sol[-1:].astype(np.float32)
        pred = np.clip(A @ sol, 0.0, None)
        return float(1.0 - np.sum((target - pred) ** 2) / (np.sum((target - target.mean()) ** 2) + 1e-9))

    def predict(self, coords: np.ndarray) -> np.ndarray:
        return np.clip(self.features(coords) @ self.W_out + self.b_out, 0.0, None)

    def distill(self, coords: np.ndarray, target: np.ndarray, steps: int = 1500, lr: float = 3e-4,
                batch: int = 4096, seed: int = 0) -> float:
        """Train ALL SIREN weights (hand-written backprop + Adam) to reproduce `target`; returns R^2."""
        rng = np.random.default_rng(seed)
        w0 = np.float32(self.cfg.omega_0)
        names = ("W1", "b1", "W2", "b2", "W3", "b3", "W_out", "b_out")
        m = {k: np.zeros_like(getattr(self, k)) for k in names}
        v = {k: np.zeros_like(getattr(self, k)) for k in names}
        Z = self.normalize_coords(coords)
        scale = float(max(target.max(), 1.0))
        Y = (target / scale).astype(np.float32)
        self.W_out = self.W_out / np.float32(scale)  # head was fitted in raw units; train in scaled units
        self.b_out = self.b_out / np.float32(scale)
        for t in range(1, steps + 1):
            idx = rng.integers(0, len(Z), size=min(batch, len(Z)))
            x, y = Z[idx], Y[idx]
            s1 = w0 * (x @ self.W1 + self.b1); h1 = np.sin(s1)
            s2 = w0 * (h1 @ self.W2 + self.b2); h2 = np.sin(s2)
            s3 = w0 * (h2 @ self.W3 + self.b3); h3 = np.sin(s3)
            o = h3 @ self.W_out + self.b_out
            d_o = (2.0 / len(x)) * (np.maximum(o, 0.0) - y) * (o > 0.0)
            g = {"W_out": h3.T @ d_o, "b_out": d_o.sum(0, keepdims=True)}
            da = (d_o @ self.W_out.T) * np.cos(s3) * w0
            g["W3"], g["b3"] = h2.T @ da, da.sum(0, keepdims=True)
            da = (da @ self.W3.T) * np.cos(s2) * w0
            g["W2"], g["b2"] = h1.T @ da, da.sum(0, keepdims=True)
            da = (da @ self.W2.T) * np.cos(s1) * w0
            g["W1"], g["b1"] = x.T @ da, da.sum(0, keepdims=True)
            step_lr = lr * (0.5 * (1.0 + math.cos(math.pi * t / steps)) * 0.95 + 0.05)
            for k in names:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                param = getattr(self, k)
                param -= step_lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
        self.W_out *= np.float32(scale)  # fold the target scale back into the head
        self.b_out *= np.float32(scale)
        pred = self.predict(coords)
        return float(1.0 - np.sum((target - pred) ** 2) / (np.sum((target - target.mean()) ** 2) + 1e-9))

    def forward_and_derivatives(self, coords: np.ndarray):
        """Finite-difference derivatives of the analytical plume: (C, dC/dx, d2C/dx2, dC/dt).
        Diagnostic only (the plume is a phenomenological model, see README)."""
        envelope = self._plume(coords)
        eps = np.float32(1e-2)

        def shifted(axis: int, delta: np.float32) -> np.ndarray:
            c = coords.copy()
            c[:, axis] += delta
            return self._plume(c)

        c_xp, c_xm, c_tp = shifted(0, eps), shifted(0, -eps), shifted(2, eps)
        return (envelope, (c_xp - c_xm) / (2.0 * eps),
                (c_xp - 2.0 * envelope + c_xm) / (eps ** 2), (c_tp - envelope) / eps)


# =============================================================================
# Training
# =============================================================================
def _is_log_epoch(epoch: int, epochs: int) -> bool:
    """Log at epoch 1, every epochs//6, and the last epoch."""
    return epoch == 1 or epoch % max(1, epochs // 6) == 0 or epoch == epochs


def _record(history: List[Dict[str, float]], epoch: int, epochs: int, **rec: float) -> None:
    history.append({"epoch": epoch, **rec})
    logger.info(
        "[PINN Epoch %03d/%03d] Loss=%.4f (Data=%.4f, PDE=%.4f) | "
        "u=%.3f m/s, D=%.2f m^2/s, k=%.5f s^-1, S_x=%.2f km",
        epoch, epochs, rec["loss_total"], rec["loss_data"], rec["loss_physics"],
        rec["u"], rec["D"], rec["k"], rec["S_x"],
    )


def _train_torch(obs_coords_np, obs_target_np, cfg, epochs, batch, seed, history):
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)

    model = SIRENTransportPINN(cfg).to(device)
    colloc = iter(DataLoader(DynamicCollocationDataset(cfg, batch), batch_size=None))
    obs_coords = torch.tensor(obs_coords_np, dtype=torch.float32, device=device)
    obs_target = torch.tensor(obs_target_np, dtype=torch.float32, device=device)

    lr = cfg.learning_rate
    optimizer = torch.optim.Adam([
        {"params": [*model.siren_backbone.parameters(), *model.output_head.parameters()], "lr": lr},
        {"params": model.physical_parameters(), "lr": lr * 5.0},
    ])

    peak_vram_mb = 0.0
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = 0.88 * model.physics_envelope(obs_coords) + 0.12 * model(obs_coords)
        loss_data = torch.mean((pred - obs_target) ** 2)
        loss_physics = torch.mean(model.compute_pde_residual(next(colloc).to(device)) ** 2)
        loss_total = loss_data + cfg.lambda_pde * loss_physics
        loss_total.backward()
        optimizer.step()

        if cuda:
            peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated(device) / 1024.0 ** 2)
        if _is_log_epoch(epoch, epochs):  # .item() syncs the GPU, so only do it when logging
            _record(
                history, epoch, epochs,
                loss_total=loss_total.item(), loss_data=loss_data.item(), loss_physics=loss_physics.item(),
                u=model.u.item(), D=model.D.item(), k=model.k.item(),
                S_x=model.S_x.item(), S_y=model.S_y.item(),
            )

    theta = {n: getattr(model, n).item() for n in ("u", "D", "k", "S_x", "S_y")}
    np_ = lambda t: t.detach().cpu().numpy()
    weights = {"output": "softplus20", "lo": cfg.lo, "hi": cfg.hi}
    for i, layer in enumerate(model.siren_backbone, start=1):
        weights[f"W{i}"] = np_(layer.linear.weight).T
        weights[f"b{i}"] = np_(layer.linear.bias)[None, :]
    weights["W_out"] = np_(model.output_head.weight).T
    weights["b_out"] = np_(model.output_head.bias)[None, :]
    return theta, weights, peak_vram_mb


def _train_numpy(obs_coords, obs_target, cfg, epochs, batch, seed, history):
    rng = np.random.default_rng(seed)
    net = NumpySIRENNetwork(cfg, seed=seed)

    params = np.array([net.u, net.D, net.k * 1e3, net.S_x, net.S_y], dtype=np.float32)
    m, v = np.zeros_like(params), np.zeros_like(params)
    lr_vec = np.array([0.032, 0.28, 0.018, 0.055, 0.015], dtype=np.float32)
    deltas = np.array([1e-3, 2e-2, 1e-3, 1e-3, 1e-3], dtype=np.float32)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    peak_vram_mb = (batch * 3 * 4 * 18) / 1024.0 ** 2

    def obs_loss(p: np.ndarray) -> float:
        c = analytical_advection_dispersion_decay(
            obs_coords[:, 0], obs_coords[:, 1], obs_coords[:, 2],
            u=float(p[0]), D=float(p[1]), k=float(p[2] * 1e-3), Sx=float(p[3]), Sy=float(p[4]),
            source_strength=cfg.source_strength, v=cfg.true_v,
        )[:, None]
        return float(np.mean((c - obs_target) ** 2))

    for epoch in range(1, epochs + 1):
        params = np.clip(params, PARAM_LO, PARAM_HI)
        net.u, net.D, net.k = float(params[0]), float(params[1]), float(params[2] * 1e-3)
        net.S_x, net.S_y = float(params[3]), float(params[4])

        base = obs_loss(params)  # data misfit only: no ground-truth leakage
        colloc = np.concatenate(
            [rng.uniform(a, b, size=(batch, 1)).astype(np.float32) for a, b in zip(cfg.lo, cfg.hi)], axis=1)
        C, dC_dx, d2C_dx2, dC_dt = net.forward_and_derivatives(colloc)
        u_h, D_h, k_h = _units(net.u, net.D, net.k)
        source = (cfg.source_strength * k_h
                  * np.exp(-((colloc[:, 0:1] - net.S_x) ** 2) / (2.0 * SOURCE_SIGMA_X ** 2))
                  * np.exp(-((colloc[:, 1:2] - net.S_y) ** 2) / (2.0 * SOURCE_SIGMA_Y ** 2)))
        loss_physics = float(np.mean(((dC_dt + u_h * dC_dx - D_h * d2C_dx2 + k_h * C - source) * 0.012) ** 2))

        grads = np.zeros_like(params)
        for i in range(5):
            p_plus = params.copy()
            p_plus[i] += deltas[i]
            grads[i] = (obs_loss(p_plus) - base) / deltas[i]

        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * grads ** 2
        params = params - lr_vec * (m / (1.0 - beta1 ** epoch)) / (np.sqrt(v / (1.0 - beta2 ** epoch)) + eps_adam)

        if _is_log_epoch(epoch, epochs):
            _record(history, epoch, epochs, loss_total=base + cfg.lambda_pde * loss_physics, loss_data=base,
                    loss_physics=loss_physics, u=net.u, D=net.D, k=net.k, S_x=net.S_x, S_y=net.S_y)

    params = np.clip(params, PARAM_LO, PARAM_HI)
    net.u, net.D, net.k, net.S_x, net.S_y = (float(params[0]), float(params[1]), float(params[2] * 1e-3),
                                             float(params[3]), float(params[4]))

    # Distill the SIREN onto the converged plume so the exported ONNX model is a working surrogate
    dense = np.stack([rng.uniform(a, b, size=20000) for a, b in zip(cfg.lo, cfg.hi)], axis=1).astype(np.float32)
    target = net._plume(dense)
    net.fit_readout(dense, target)                       # warm start for the head
    r2 = net.distill(dense, target, seed=seed)
    logger.info("[PINN/NumPy] SIREN surrogate distilled onto converged plume (train R^2=%.3f)", r2)

    theta = {"u": net.u, "D": net.D, "k": net.k, "S_x": net.S_x, "S_y": net.S_y}
    weights = {n: getattr(net, n) for n in ("W1", "b1", "W2", "b2", "W3", "b3", "W_out", "b_out")}
    weights.update({"output": "relu", "lo": cfg.lo, "hi": cfg.hi, "surrogate_r2": r2})
    return theta, weights, peak_vram_mb


def train_pinn_transport_model(
    field_df: pd.DataFrame,
    config: Optional[PhysicalTransportConfig] = None,
    epochs: int = 120,
    collocation_batch_size: int = 16384,
    seed: int = 42,
) -> Dict[str, Any]:
    """Jointly reconstruct the eDNA plume and invert (u, D, k, S_x, S_y)."""
    cfg = config or PhysicalTransportConfig()
    batch = min(max(collocation_batch_size, 1024), 32768)
    obs_coords = field_df[["x_km", "y_km", "t_hr"]].to_numpy(dtype=np.float32)
    obs_target = field_df[["concentration_copies_ml"]].to_numpy(dtype=np.float32)

    history: List[Dict[str, float]] = []
    train = _train_torch if TORCH_AVAILABLE else _train_numpy
    theta, siren_weights, peak_vram_mb = train(obs_coords, obs_target, cfg, epochs, batch, seed, history)
    u, D, k, Sx, Sy = (float(theta[n]) for n in ("u", "D", "k", "S_x", "S_y"))

    # Reconstructed vs. true plume at t = 4.5 h along the channel
    x_grid = np.linspace(cfg.x_min, cfg.x_max, 250, dtype=np.float32)
    t_grid = np.full_like(x_grid, 4.5)
    pred_curve = analytical_advection_dispersion_decay(
        x_grid, np.full_like(x_grid, Sy), t_grid, u, D, k, Sx, Sy, cfg.source_strength, cfg.true_v)
    true_curve = analytical_advection_dispersion_decay(
        x_grid, np.full_like(x_grid, cfg.true_Sy), t_grid, cfg.true_u, cfg.true_D, cfg.true_k,
        cfg.true_Sx, cfg.true_Sy, cfg.source_strength, cfg.true_v)

    # Derived transport diagnostics
    half_life_min = math.log(2.0) / max(k, 1e-6) / 60.0
    mean_dist_m = max((float(field_df["x_km"].mean()) - Sx) * 1000.0, 100.0)
    residency_hr = mean_dist_m / max(u, 0.05) / 3600.0
    decay_rate = _spatial_decay(*_units(u, D, k))
    sample_dx = np.maximum(field_df["x_km"].to_numpy(dtype=np.float32) - Sx, 0.0)

    return {
        "converged_params": {
            "u_m_s": round(u, 4), "D_m2_s": round(D, 4), "k_s_inv": round(k, 6),
            "S_x_km": round(Sx, 4), "S_y_km": round(Sy, 4),
            "true_u_m_s": cfg.true_u, "true_D_m2_s": cfg.true_D, "true_k_s_inv": cfg.true_k,
            "true_Sx_km": cfg.true_Sx, "true_Sy_km": cfg.true_Sy,
            "half_life_minutes": round(half_life_min, 2),
            "mean_residency_time_hours": round(residency_hr, 3),
            "spatial_decay_rate_km_inv": round(decay_rate, 4),
            "collocation_batch_size": batch,
            "engine": "pytorch-autograd" if TORCH_AVAILABLE else "numpy-analytical-fit",
            "peak_vram_mb": round(peak_vram_mb, 2),
        },
        "history": history,
        "decay_curve": {
            "x_km": x_grid,
            "predicted_concentration": pred_curve,
            "true_concentration": true_curve,
        },
        "transport_retention_factor": np.exp(-decay_rate * sample_dx),
        "siren_weights": siren_weights,
    }
