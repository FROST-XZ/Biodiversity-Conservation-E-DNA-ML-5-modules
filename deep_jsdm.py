"""
Module 3: Low-Rank Deep Joint Species Distribution Model & Explainable AI
=========================================================================
Links transport-corrected presence-absence data with 5 spatial covariates:
canopy height (LiDAR), NDVI, DEM elevation, terrain slope and TWI.

- Covariates are sampled from GeoTIFFs via `rasterio.sample` (out-of-core) when
  present, otherwise synthesized along the 25 km floodplain reach.
- Regularized MLP (L1 + L2 + dropout) maps covariates to species logits.
- Residual co-occurrence uses a low-rank factorization Sigma = L L^T, L in R^(S x d),
  so memory scales O(S d) instead of O(S^2).
- xAI: permutation feature importance (Delta MAE in community composition).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import rasterio  # type: ignore

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("eDNA.DeepJSDM")

COVARIATE_COLUMNS: List[str] = [
    "lidar_canopy_height_m",
    "multispectral_ndvi",
    "elevation_dem_m",
    "terrain_slope_deg",
    "topographic_wetness_index",
]
COVARIATE_LABELS: Dict[str, str] = {
    "topographic_wetness_index": "Topographic Wetness Index (TWI)",
    "lidar_canopy_height_m": "LiDAR 3D Canopy Height (m)",
    "multispectral_ndvi": "Multispectral NDVI (Sentinel-2)",
    "elevation_dem_m": "Digital Elevation Model (DEM, m)",
    "terrain_slope_deg": "Terrain Slope Gradient (deg)",
}
GEOTIFF_FILES = {
    "lidar_canopy_height_m": "lidar_canopy_height.tif",
    "multispectral_ndvi": "sentinel2_ndvi.tif",
    "elevation_dem_m": "srtm_dem.tif",
    "terrain_slope_deg": "terrain_slope.tif",
    "topographic_wetness_index": "twi.tif",
}

# Species x covariate response coefficients, in COVARIATE_COLUMNS order
# [Canopy, NDVI, DEM, Slope, TWI]; rows follow the 12 reference taxa (SP01-SP12).
TRUE_BETA = np.array(
    [
        [0.65, 0.80, -0.55, -0.45, 1.25],  # Platanista gangetica (deep wet channels)
        [0.45, 0.35, 1.15, 0.75, -0.40],   # Tor putitora (upland riffles)
        [1.10, 1.05, -0.35, -0.60, 1.30],  # Channa barca (vegetated floodplain wetlands)
        [0.30, 0.55, -0.40, -0.50, 1.10],  # Nilssonia nigricans (backwater sandbars)
        [1.25, 0.95, 0.20, 0.30, 0.75],    # Lutrogale perspicillata (riparian canopy banks)
        [0.40, 0.70, -0.45, -0.35, 0.95],  # Labeo rohita
        [0.35, 0.65, -0.50, -0.40, 0.90],  # Catla catla
        [0.20, 0.45, -0.90, -0.55, 1.05],  # Tenualosa ilisha (lower reach migratory)
        [0.50, 0.60, -0.60, -0.30, 1.15],  # Wallago attu
        [0.75, 0.85, -0.55, -0.45, 1.20],  # Macrobrachium rosenbergii
        [0.60, 0.50, 0.85, 0.65, 0.25],    # Botia dario
        [0.25, 0.40, -0.30, 0.45, 0.85],   # Gavialis gangeticus
    ],
    dtype=np.float32,
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def query_geotiff_covariates(
    coords_xy: np.ndarray,
    data_dir: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Covariates from `{data_dir}/orang_mangaldoi_remote_sensing_covariates.csv` (row-aligned with the
    field samples), else GeoTIFFs in `data_dir`, else synthetic floodplain gradients."""
    if data_dir:
        cov_csv = os.path.join(data_dir, "orang_mangaldoi_remote_sensing_covariates.csv")
        if os.path.isfile(cov_csv):
            df_cov = pd.read_csv(cov_csv)
            if set(COVARIATE_COLUMNS) <= set(df_cov.columns) and len(df_cov) == len(coords_xy):
                logger.info("Loading remote-sensing covariates from %s (matched to samples BY ROW ORDER)", cov_csv)
                return df_cov[COVARIATE_COLUMNS].copy()
            logger.warning("Covariate CSV %s ignored: missing columns or %d rows != %d samples",
                           cov_csv, len(df_cov), len(coords_xy))
    if data_dir and RASTERIO_AVAILABLE:
        paths = {c: os.path.join(data_dir, f) for c, f in GEOTIFF_FILES.items()}
        if all(os.path.isfile(p) for p in paths.values()):
            logger.info("Streaming point queries from GeoTIFF rasters in %s", data_dir)
            points = [(float(x), float(y)) for x, y in coords_xy]
            out = {}
            for col, path in paths.items():
                with rasterio.open(path) as src:
                    out[col] = np.asarray([float(v[0]) for v in src.sample(points)], dtype=np.float32)
            return pd.DataFrame(out)

    rng = np.random.default_rng(seed)
    n = len(coords_xy)
    x, y = coords_xy[:, 0], coords_xy[:, 1]
    noise = lambda s: rng.normal(0.0, s, size=n)

    # Draw order (dem, slope, twi, ndvi, canopy) is fixed for reproducibility
    dem = (67.5 - 0.62 * x + 3.2 * np.abs(y - 0.55) + noise(0.95)).clip(48.0, 74.0)
    slope = (1.8 + 4.5 * np.abs(y - 0.50) + 1.4 * np.sin(x * 0.45) ** 2 + noise(0.45)).clip(0.3, 12.5)
    twi = (14.8 - 0.38 * slope - 0.08 * (dem - 55.0) + 2.4 * np.cos(x * 0.28) + noise(0.65)).clip(7.5, 19.5)
    ndvi = (0.42 + 0.025 * (twi - 12.0) + 0.18 * np.sin(x * 0.22 + 0.5) + noise(0.055)).clip(0.08, 0.86)
    canopy = (8.5 + 18.0 * np.maximum(ndvi - 0.25, 0.0) + 0.45 * (twi - 12.0) + noise(1.35)).clip(0.5, 29.5)

    return pd.DataFrame(
        {
            "lidar_canopy_height_m": np.round(canopy, 3),
            "multispectral_ndvi": np.round(ndvi, 4),
            "elevation_dem_m": np.round(dem, 2),
            "terrain_slope_deg": np.round(slope, 3),
            "topographic_wetness_index": np.round(twi, 3),
        }
    )


if TORCH_AVAILABLE:

    class LowRankDeepJSDM(nn.Module):
        """Covariates -> habitat embedding -> species probabilities, with Sigma = L L^T."""

        def __init__(self, num_covariates: int = 5, num_species: int = 12, hidden_dim: int = 32,
                     latent_rank: int = 4, dropout_p: float = 0.15) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(num_covariates, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.species_head = nn.Linear(hidden_dim, num_species)       # environmental fixed effects
            self.site_latent_head = nn.Linear(hidden_dim, latent_rank)   # site factors z(x)
            self.species_loadings_L = nn.Parameter(torch.randn(num_species, latent_rank) * 0.15)

        def forward(self, x_cov: torch.Tensor) -> torch.Tensor:
            h = self.encoder(x_cov)
            co_logits = self.site_latent_head(h) @ self.species_loadings_L.t()
            return torch.sigmoid(self.species_head(h) + co_logits)

        def get_low_rank_covariance(self) -> np.ndarray:
            L = self.species_loadings_L.detach().cpu().numpy()
            return (L @ L.T).astype(np.float32)


class NumpyLowRankDeepJSDM:
    """FP32 NumPy MLP + low-rank latent factors with L1/L2 decay and inverted dropout."""

    def __init__(self, num_covariates: int = 5, num_species: int = 12, hidden_dim: int = 32,
                 latent_rank: int = 4, dropout_p: float = 0.15, seed: int = 42) -> None:
        self.dropout_p = dropout_p
        self.rng = np.random.default_rng(seed)
        normal = lambda scale, shape: self.rng.normal(0.0, scale, size=shape).astype(np.float32)
        zeros = lambda cols: np.zeros((1, cols), dtype=np.float32)

        # Draw order matters for reproducibility
        self.W1, self.b1 = normal(np.sqrt(2.0 / num_covariates), (num_covariates, hidden_dim)), zeros(hidden_dim)
        self.W2, self.b2 = normal(np.sqrt(2.0 / hidden_dim), (hidden_dim, hidden_dim)), zeros(hidden_dim)
        self.W_sp, self.b_sp = normal(np.sqrt(1.0 / hidden_dim), (hidden_dim, num_species)), zeros(num_species)
        self.W_z = normal(np.sqrt(1.0 / hidden_dim), (hidden_dim, latent_rank))
        self.L = normal(0.20, (num_species, latent_rank))

    def forward(self, x: np.ndarray, training: bool = False) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(0.0, z1)
        if training and self.dropout_p > 0.0:
            keep = (self.rng.random(h1.shape) >= self.dropout_p).astype(np.float32)
            h1 = h1 * (keep / (1.0 - self.dropout_p))
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0.0, z2)
        site_z = h2 @ self.W_z
        probs = _sigmoid(h2 @ self.W_sp + self.b_sp + site_z @ self.L.T).astype(np.float32)
        return probs, {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "site_z": site_z}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float = 0.035,
            l1: float = 1e-4, l2: float = 5e-4) -> None:
        """Full-batch gradient descent on MSE-through-sigmoid with L1/L2 and step decay."""
        n = len(X)
        for epoch in range(1, epochs + 1):
            probs, c = self.forward(X, training=True)
            diff = (probs - Y) / n

            dsite_z = diff @ self.L
            dz2 = (diff @ self.W_sp.T + dsite_z @ self.W_z.T) * (c["z2"] > 0).astype(np.float32)
            dz1 = (dz2 @ self.W2.T) * (c["z1"] > 0).astype(np.float32)
            grads = {
                "W_sp": c["h2"].T @ diff + l2 * self.W_sp + l1 * np.sign(self.W_sp),
                "b_sp": diff.sum(axis=0, keepdims=True),
                "L": diff.T @ c["site_z"] + l2 * self.L,
                "W_z": c["h2"].T @ dsite_z + l2 * self.W_z,
                "W2": c["h1"].T @ dz2 + l2 * self.W2 + l1 * np.sign(self.W2),
                "b2": dz2.sum(axis=0, keepdims=True),
                "W1": c["x"].T @ dz1 + l2 * self.W1 + l1 * np.sign(self.W1),
                "b1": dz1.sum(axis=0, keepdims=True),
            }
            step = lr * (0.985 ** (epoch // 15))
            for name, g in grads.items():
                param = getattr(self, name)
                param -= step * g  # in-place update

    def get_low_rank_covariance(self) -> np.ndarray:
        return (self.L @ self.L.T).astype(np.float32)


def _train_torch(X: np.ndarray, Y: np.ndarray, rank: int, epochs: int, seed: int,
                 l1: float, l2: float) -> Tuple[np.ndarray, np.ndarray, Callable]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LowRankDeepJSDM(X.shape[1], Y.shape[1], hidden_dim=32, latent_rank=rank, dropout_p=0.12).to(device)
    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=l2)
    bce = nn.BCELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        l1_pen = sum(p.abs().sum() for p in model.parameters())
        (bce(model(x_t), y_t) + l1 * l1_pen).backward()
        optimizer.step()

    model.eval()

    def predict(x_in: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.tensor(x_in, dtype=torch.float32, device=device)).cpu().numpy()

    return predict(X), model.get_low_rank_covariance(), predict


def _train_numpy(X: np.ndarray, Y: np.ndarray, rank: int, epochs: int, seed: int,
                 l1: float, l2: float) -> Tuple[np.ndarray, np.ndarray, Callable]:
    net = NumpyLowRankDeepJSDM(X.shape[1], Y.shape[1], hidden_dim=32, latent_rank=rank,
                               dropout_p=0.10, seed=seed)
    net.fit(X, Y, epochs, lr=0.035, l1=l1, l2=l2)
    return net.predict(X), net.get_low_rank_covariance(), net.predict


def _permutation_importance(predict: Callable, X: np.ndarray, Y: np.ndarray, baseline_mae: float,
                            rng: np.random.Generator, n_repeats: int = 25) -> List[Dict[str, Any]]:
    """Mean/std Delta MAE after shuffling each covariate across sites, ranked descending."""
    rows = []
    for j, col in enumerate(COVARIATE_COLUMNS):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            deltas.append(max(float(np.mean(np.abs(predict(X_perm) - Y))) - baseline_mae, 0.0))
        rows.append({
            "covariate": col,
            "label": COVARIATE_LABELS[col],
            "delta_mae": round(float(np.mean(deltas)), 5),
            "delta_mae_std": round(float(np.std(deltas)), 5),
        })

    rows.sort(key=lambda r: r["delta_mae"], reverse=True)
    total = sum(r["delta_mae"] for r in rows) + 1e-8
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank
        r["relative_importance_pct"] = round(100.0 * r["delta_mae"] / total, 2)
    return rows


def run_deep_jsdm_pipeline(
    field_df: pd.DataFrame,
    qc_results: Dict[str, Any],
    pinn_results: Dict[str, Any],
    export_dir: str,
    data_dir: Optional[str] = None,
    latent_rank: int = 4,
    epochs: int = 160,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build covariates, train the JSDM, rank drivers by permutation importance, export CSV."""
    rng = np.random.default_rng(seed)
    cov_df = query_geotiff_covariates(field_df[["x_km", "y_km"]].to_numpy(dtype=np.float32),
                                      data_dir=data_dir, seed=seed)
    X_raw = cov_df[COVARIATE_COLUMNS].to_numpy(dtype=np.float32)
    x_mean = X_raw.mean(axis=0, keepdims=True)
    x_std = X_raw.std(axis=0, keepdims=True) + 1e-6
    X_norm = (X_raw - x_mean) / x_std

    # Target: habitat suitability blended with transport-weighted metagenomic detections
    raw_pa = qc_results["presence_absence_matrix"].astype(np.float32)
    n_samples, n_species = raw_pa.shape
    retention = pinn_results["transport_retention_factor"][:, None].astype(np.float32)
    env_probs = 1.0 / (1.0 + np.exp(-(X_norm @ TRUE_BETA.T)))
    Y_target = np.clip(0.72 * env_probs + 0.28 * raw_pa * np.clip(retention + 0.35, 0.5, 1.0),
                       0.0, 1.0).astype(np.float32)
    Y_binary = (Y_target >= 0.50).astype(np.int32)

    # Transport-corrected presence-absence goes to its OWN file (the QC matrix stays untouched)
    corrected_pa = qc_results["pa_df"].copy()
    for idx, taxon in enumerate(qc_results["taxa_metadata"]):
        corrected_pa[taxon["scientific_name"]] = Y_binary[:, idx]
    os.makedirs(export_dir, exist_ok=True)
    corrected_pa_path = os.path.join(export_dir, "transport_corrected_presence_absence.csv")
    corrected_pa.to_csv(corrected_pa_path, index=False)

    # Hold-out split: metrics and permutation importance are computed on unseen sites only
    perm = rng.permutation(n_samples)
    n_test = max(int(round(0.25 * n_samples)), 1)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    train = _train_torch if TORCH_AVAILABLE else _train_numpy
    _, low_rank_cov, predict = train(X_norm[train_idx], Y_target[train_idx], latent_rank, epochs, seed, 1e-4, 5e-4)
    pred_probs = predict(X_norm)

    r2 = lambda obs, pred: float(1.0 - np.sum((obs - pred) ** 2) / (np.sum((obs - obs.mean()) ** 2) + 1e-8))
    abs_err = np.abs(pred_probs - Y_target)
    train_mae, baseline_mae = float(abs_err[train_idx].mean()), float(abs_err[test_idx].mean())
    perm_importance = _permutation_importance(predict, X_norm[test_idx], Y_target[test_idx], baseline_mae, rng)

    observed = Y_binary.sum(axis=1)
    predicted = pred_probs.sum(axis=1)
    richness_r2 = r2(observed[test_idx], predicted[test_idx])
    train_richness_r2 = r2(observed[train_idx], predicted[train_idx])

    base_cols = ["sample_id", "reach_zone"] + [c for c in ("lat_deg", "lon_deg") if c in field_df.columns]
    env_out_df = field_df[base_cols + ["x_km", "y_km"]].join(cov_df[COVARIATE_COLUMNS])
    env_out_df["observed_species_richness"] = observed
    env_out_df["predicted_species_richness"] = np.round(predicted, 3)
    env_out_df["community_mae"] = np.round(abs_err.mean(axis=1), 4)

    env_out_df["split"] = np.where(np.isin(np.arange(n_samples), test_idx), "holdout", "train")
    env_csv_path = os.path.join(export_dir, "environmental_covariates_and_richness.csv")
    env_out_df.to_csv(env_csv_path, index=False)

    logger.info(
        "[Deep JSDM & xAI] Low-rank factor rank d=%d (S=%d taxa, memory saving %.1fx vs dense). "
        "Hold-out MAE=%.4f (train %.4f), hold-out richness R^2=%.4f (train %.4f). Top driver: %s (Delta MAE=+%.4f)",
        latent_rank, n_species, n_species / latent_rank, baseline_mae, train_mae, richness_r2, train_richness_r2,
        perm_importance[0]["label"], perm_importance[0]["delta_mae"],
    )

    return {
        "env_df": env_out_df,
        "perm_importance": perm_importance,
        "baseline_mae": round(baseline_mae, 4),
        "richness_r2": round(richness_r2, 4),
        "train_mae": round(train_mae, 4),
        "train_richness_r2": round(train_richness_r2, 4),
        "n_holdout_sites": int(n_test),
        "corrected_pa_csv_path": corrected_pa_path,
        "low_rank_covariance": low_rank_cov,
        "predicted_probabilities": pred_probs,
        "env_csv_path": env_csv_path,
        "x_mean": x_mean,
        "x_std": x_std,
    }
