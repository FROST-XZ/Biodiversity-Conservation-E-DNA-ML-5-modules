"""
Module 4: Taxonomy-Free Supervised ML for Dark Diversity & Biotic Indices
=========================================================================
Predicts Ecological Quality Status (EQS in [0, 1]) directly from sparse, zero-inflated
ASV count tables, without species-level labels ("dark diversity").

1. Compositional preprocessing: relative abundance + pseudocount Centered Log-Ratio (CLR).
2. Regularized deep regressor (LayerNorm + ReLU + Sigmoid head); PyTorch when available,
   otherwise a NumPy implementation.
3. K-fold cross-validation (MSE, R^2) and export of `asv_relative_abundance_and_health.csv`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("eDNA.TaxonomyFreeML")


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1.0 - np.sum((y - pred) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-8))


def _scaled_minmax(a: np.ndarray, weight: float = 1.0) -> np.ndarray:
    return weight * (a - np.min(a)) / (np.ptp(a) + 1e-6)


def preprocess_asv_compositional(
    asv_counts: np.ndarray,
    pseudocount: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (relative abundance, CLR features): CLR(x_i) = ln(p_i) - mean_j ln(p_j)."""
    counts = np.maximum(asv_counts.astype(np.float32), 0.0)
    totals = counts.sum(axis=1, keepdims=True)
    rel_abundance = (counts / np.where(totals <= 0.0, 1.0, totals)).astype(np.float32)
    log_counts = np.log(counts + pseudocount)
    clr = (log_counts - log_counts.mean(axis=1, keepdims=True)).astype(np.float32)
    return rel_abundance, clr


def simulate_or_load_asv_table(
    field_df: pd.DataFrame,
    env_df: pd.DataFrame,
    num_asvs: int = 250,
    data_dir: Optional[str] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (counts, eqs, ambi, shannon) from `{data_dir}/asv_counts.csv` or a simulation."""
    if data_dir:
        csv = os.path.join(data_dir, "asv_counts.csv")
        if os.path.isfile(csv):
            logger.info("Loading ASV count table from %s", csv)
            df = pd.read_csv(csv)
            asv_cols = [c for c in df.columns if c.startswith("ASV_")]
            if asv_cols and "eqs_ground_truth" in df.columns and len(df) == len(field_df):
                eqs = df["eqs_ground_truth"].to_numpy(dtype=np.float32)
                ambi = df.get("ambi_score", pd.Series(6.0 * (1.0 - eqs))).to_numpy(dtype=np.float32)
                shannon = df.get("shannon_diversity", pd.Series(2.5 + 1.8 * eqs)).to_numpy(dtype=np.float32)
                return df[asv_cols].to_numpy(dtype=np.float32), eqs, ambi, shannon

    rng = np.random.default_rng(seed)
    n = len(field_df)
    col = lambda name: env_df[name].to_numpy(dtype=np.float32)

    # Latent ecological quality driven by habitat covariates
    raw_health = (
        _scaled_minmax(col("multispectral_ndvi"), 0.38)
        + _scaled_minmax(col("lidar_canopy_height_m"), 0.32)
        + _scaled_minmax(col("topographic_wetness_index"), 0.20)
        - _scaled_minmax(col("terrain_slope_deg"), 0.10)
    )
    raw_health = _scaled_minmax(raw_health)
    eqs = np.clip(0.12 + 0.82 * raw_health + rng.normal(0.0, 0.035, size=n), 0.05, 0.98)

    # Guilds: sensitive (+EQS) -> tolerant -> opportunistic (-EQS)
    sensitivity = np.linspace(1.6, -1.6, num_asvs, dtype=np.float32)
    base_prevalence = rng.uniform(0.22, 0.55, size=(1, num_asvs)).astype(np.float32)
    affinity = 1.0 / (1.0 + np.exp(-3.2 * ((eqs.astype(np.float32)[:, None] - 0.52) * sensitivity[None, :])))

    # Zero inflation (~60-68% sparsity) and log-normal abundance bursts
    active_prob = np.clip(base_prevalence * (0.45 + 1.15 * affinity), 0.05, 0.85)
    present = (rng.random((n, num_asvs)) < active_prob).astype(np.float32)
    rate = np.exp(4.8 * affinity + rng.normal(0.0, 0.35, size=(n, num_asvs)))
    counts = np.round(present * rate).astype(np.float32)
    counts[counts.sum(axis=1) == 0, :5] = 15.0  # every sample needs reads

    # Refine EQS with the realized guild composition
    rel, _ = preprocess_asv_compositional(counts)
    realized = _scaled_minmax(rel @ sensitivity)
    eqs = np.clip(0.10 + 0.85 * (0.75 * realized + 0.25 * raw_health) + rng.normal(0.0, 0.02, size=n),
                  0.06, 0.97).astype(np.float32)

    # Shannon H' = -sum p ln p, and AMBI score
    log_p = np.where(rel > 1e-8, np.log(np.maximum(rel, 1e-8)), 0.0)
    shannon = (-np.sum(rel * log_p, axis=1)).astype(np.float32)
    ambi = np.clip(5.8 * (1.0 - eqs) + rng.normal(0.0, 0.12, size=n), 0.2, 5.9).astype(np.float32)
    return counts, eqs, ambi, shannon


if TORCH_AVAILABLE:

    class TaxonomyFreeDeepRegressor(nn.Module):
        """LayerNorm + ReLU MLP with a sigmoid head mapping ASV profiles to EQS."""

        def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(p=0.15),
                nn.Linear(hidden_dim, 32), nn.LayerNorm(32), nn.ReLU(),
                nn.Linear(32, 1), nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)


class NumpyTaxonomyFreeRegressor:
    """NumPy FP32 one-hidden-layer LayerNorm + ReLU + Sigmoid regressor trained with Adam."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.gamma1 = np.ones((1, hidden_dim), dtype=np.float32)
        self.beta1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, 1)).astype(np.float32)
        self.b2 = np.zeros((1, 1), dtype=np.float32)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        z1 = X @ self.W1 + self.b1
        std1 = np.sqrt(np.var(z1, axis=-1, keepdims=True) + 1e-5)
        xhat = (z1 - np.mean(z1, axis=-1, keepdims=True)) / std1
        ln1 = self.gamma1 * xhat + self.beta1
        h1 = np.maximum(0.0, ln1)
        out = (1.0 / (1.0 + np.exp(-np.clip(h1 @ self.W2 + self.b2, -20.0, 20.0)))).astype(np.float32)
        return out.squeeze(-1), {"X": X, "xhat": xhat, "ln1": ln1, "std1": std1, "h1": h1, "out": out}

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 180, lr: float = 0.025, l2: float = 1e-3) -> None:
        n, y_col = len(X), y[:, None].astype(np.float32)
        names = ("W1", "b1", "gamma1", "beta1", "W2", "b2")
        m = {k: np.zeros_like(getattr(self, k)) for k in names}
        v = {k: np.zeros_like(getattr(self, k)) for k in names}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for t in range(1, epochs + 1):
            _, c = self.forward(X)
            out = c["out"]
            dz2 = (2.0 / n) * (out - y_col) * out * (1.0 - out)  # MSE through sigmoid
            dln1 = (dz2 @ self.W2.T) * (c["ln1"] > 0).astype(np.float32)
            dxhat = dln1 * self.gamma1  # exact LayerNorm backward
            dz1 = (dxhat - dxhat.mean(axis=-1, keepdims=True)
                   - c["xhat"] * (dxhat * c["xhat"]).mean(axis=-1, keepdims=True)) / c["std1"]
            grads = {
                "W1": c["X"].T @ dz1 + l2 * self.W1, "b1": dz1.sum(axis=0, keepdims=True),
                "gamma1": (dln1 * c["xhat"]).sum(axis=0, keepdims=True), "beta1": dln1.sum(axis=0, keepdims=True),
                "W2": c["h1"].T @ dz2 + l2 * self.W2, "b2": dz2.sum(axis=0, keepdims=True),
            }
            for k, g in grads.items():
                m[k][:] = beta1 * m[k] + (1.0 - beta1) * g
                v[k][:] = beta2 * v[k] + (1.0 - beta2) * g ** 2
                param = getattr(self, k)
                param -= lr * (m[k] / (1.0 - beta1 ** t)) / (np.sqrt(v[k] / (1.0 - beta2 ** t)) + eps)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]


def _fit_predict_fold(X_tr, y_tr, X_va, seed):
    """Train one fold; returns (validation predictions, exported weights)."""
    dim = X_tr.shape[1]
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TaxonomyFreeDeepRegressor(dim, hidden_dim=64).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=6e-3, weight_decay=1e-3)
        x_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
        model.train()
        for _ in range(150):
            opt.zero_grad()
            nn.functional.mse_loss(model(x_t), y_t).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(X_va, dtype=torch.float32, device=device)).cpu().numpy(), {}

    reg = NumpyTaxonomyFreeRegressor(dim, hidden_dim=64, seed=seed)
    reg.fit(X_tr, y_tr, epochs=160, lr=0.018, l2=2e-3)
    return reg.predict(X_va), {"W1": reg.W1, "b1": reg.b1, "W2": reg.W2, "b2": reg.b2}


def run_taxonomy_free_ml_pipeline(
    field_df: pd.DataFrame,
    env_df: pd.DataFrame,
    export_dir: str,
    num_asvs: int = 250,
    data_dir: Optional[str] = None,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate/load ASVs, run K-fold CV for EQS prediction, and export the CSV."""
    rng = np.random.default_rng(seed)
    counts, eqs, ambi, shannon = simulate_or_load_asv_table(field_df, env_df, num_asvs, data_dir, seed)
    rel, clr = preprocess_asv_compositional(counts)
    sparsity_pct = float(100.0 * np.mean(counts == 0))
    X = np.concatenate([rel * 10.0, clr], axis=1).astype(np.float32)

    n = len(field_df)
    indices = np.arange(n)
    rng.shuffle(indices)
    oof = np.zeros(n, dtype=np.float32)
    fold_metrics: List[Dict[str, float]] = []
    last_weights: Dict[str, np.ndarray] = {}

    for fold, val_idx in enumerate(np.array_split(indices, n_folds)):
        train_idx = np.setdiff1d(indices, val_idx)
        X_tr, X_va = X[train_idx], X[val_idx]
        mean, std = X_tr.mean(axis=0, keepdims=True), X_tr.std(axis=0, keepdims=True) + 1e-5  # no leakage
        preds, weights = _fit_predict_fold((X_tr - mean) / std, eqs[train_idx], (X_va - mean) / std, seed + fold)
        last_weights = weights or last_weights
        oof[val_idx] = preds
        y_va = eqs[val_idx]
        fold_metrics.append({
            "fold": fold + 1,
            "mse": round(float(np.mean((preds - y_va) ** 2)), 5),
            "r2": round(_r2(y_va, preds), 4),
        })

    cv_mse = float(np.mean((oof - eqs) ** 2))
    cv_r2 = _r2(eqs, oof)

    # WFD Ecological Quality Status bands
    status = np.select([oof >= 0.80, oof >= 0.60, oof >= 0.40, oof >= 0.20],
                       ["High", "Good", "Moderate", "Poor"], default="Bad")

    base_cols = ["sample_id", "reach_zone"] + [c for c in ("lat_deg", "lon_deg") if c in field_df.columns]
    asv_df = field_df[base_cols + ["x_km", "y_km"]].copy()
    asv_df["total_asv_reads"] = counts.sum(axis=1).astype(np.int64)
    asv_df["active_asv_richness"] = (counts > 0).sum(axis=1).astype(np.int32)
    asv_df["shannon_diversity_h"] = np.round(shannon, 4)
    asv_df["ambi_biotic_score"] = np.round(ambi, 4)
    asv_df["eqs_ground_truth"] = np.round(eqs, 4)
    asv_df["eqs_predicted_cv"] = np.round(oof, 4)
    asv_df["eqs_residual"] = np.round(oof - eqs, 4)
    asv_df["ecological_status_class"] = status
    for idx in np.argsort(rel.mean(axis=0))[::-1][:25]:  # 25 most abundant ASVs
        asv_df[f"ASV_{idx + 1:04d}_rel_abund"] = np.round(rel[:, idx], 5)

    os.makedirs(export_dir, exist_ok=True)
    asv_csv_path = os.path.join(export_dir, "asv_relative_abundance_and_health.csv")
    asv_df.to_csv(asv_csv_path, index=False)

    logger.info(
        "[Taxonomy-Free ML] Evaluated %d ASVs (%.1f%% zero-inflated) via %d-fold CV. "
        "Out-of-fold R^2=%.4f, MSE=%.5f",
        num_asvs, sparsity_pct, n_folds, cv_r2, cv_mse,
    )

    return {
        "asv_df": asv_df,
        "cv_mse": round(cv_mse, 5),
        "cv_r2": round(cv_r2, 4),
        "fold_metrics": fold_metrics,
        "eqs_ground_truth": eqs,
        "eqs_predicted_cv": oof,
        "shannon_diversity": shannon,
        "ambi_score": ambi,
        "sparsity_pct": round(sparsity_pct, 2),
        "num_asvs": num_asvs,
        "asv_csv_path": asv_csv_path,
        "trained_weights": last_weights,
    }
