#!/usr/bin/env python3
"""
Unified CLI Orchestrator: End-to-End Physics-Informed & Metagenomic eDNA ML Framework
====================================================================================
Demonstrated on the Brahmaputra River Basin (Orang-Kharupetia-Mangaldoi Braided Reach).
Optimized for an NVIDIA RTX 4050 (6 GB VRAM, Ada Lovelace Tensor Cores) + 16 GB System RAM
with Crash-Proof Hardware Guardrails:
  1. Reserves 2 CPU cores for Windows/OS responsiveness and uses all remaining cores
     for parallel matrix & tensor operations (`torch.set_num_threads`).
  2. Caps CUDA VRAM allocation at 75% (4.50 GB of 6.00 GB on RTX 4050) so the display
     driver, browser, and PowerPoint never freeze or crash.
  3. Enables Ada Lovelace TF32 Tensor Cores (`allow_tf32 = True`) and scales PDE
     collocation batches to 16,384 points per step on GPU.
  4. Enforces proactive garbage collection (`gc.collect()`) and `torch.cuda.empty_cache()`
     between modules to keep peak system RAM well below the 16 GB limit.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from typing import Any, Dict

try:
    import resource  # Unix/Linux/macOS only
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Configure multi-core CPU threading before importing heavy math libraries (leave 2 cores for OS UI)
TOTAL_CPU_CORES = os.cpu_count() or 8
SAFE_WORKER_THREADS = max(1, TOTAL_CPU_CORES - 2)
for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(env_var, str(SAFE_WORKER_THREADS))

from bioinformatics_qc import run_bioinformatics_qc_pipeline
from deep_jsdm import run_deep_jsdm_pipeline
from export_and_viz import (
    export_open_edna_onnx,
    generate_interactive_html_dashboard,
    generate_visual_dashboard,
)
from pinn_transport import (
    TORCH_AVAILABLE,
    PhysicalTransportConfig,
    load_real_data,
    train_pinn_transport_model,
)
from taxonomy_free_ml import run_taxonomy_free_ml_pipeline

if TORCH_AVAILABLE:
    import torch
    torch.set_num_threads(SAFE_WORKER_THREADS)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

VRAM_TOTAL_RTX4050_MB = 6144.0
VRAM_SAFE_CEILING_MB = 4608.0   # 75% of 6 GB RTX 4050 VRAM (leaves 1.5 GB for Windows OS & display)
RAM_TOTAL_SYSTEM_MB = 16384.0
RAM_SAFE_CEILING_MB = 12288.0   # 75% of 16 GB System RAM (leaves 4 GB for OS & apps)

CLI_ARGS = [
    ("--epochs_pinn", int, 120, "Training epochs for the SIREN PINN inverse transport module"),
    ("--samples", int, 160, "Number of field eDNA samples to ingest or evaluate"),
    ("--asvs", int, 250, "Number of taxonomy-free ASVs"),
    ("--batch_size", int, 16384, "PDE collocation batch size for RTX 4050 (clamped to 1024-32768)"),
    ("--vram_fraction", float, 0.75, "Max fraction of 6 GB RTX 4050 VRAM to allocate (default: 0.75 = 4.5 GB)"),
    ("--data_dir", str, DEFAULT_DATA_DIR, "Directory with reference-calibrated .csv, .fastq.gz, and .tif datasets"),
    ("--export_dir", str, SCRIPT_DIR, "Output dir for CSVs, ONNX model, PNG and HTML"),
    ("--jsdm_epochs", int, 160, "Training epochs for the deep JSDM"),
    ("--seed", int, 42, "Random seed for reproducibility"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-End Physics-Informed & Metagenomic eDNA ML Framework (RTX 4050 6GB + 16GB RAM Edition)"
    )
    for flag, typ, default, help_ in CLI_ARGS:
        parser.add_argument(flag, type=typ, default=default, help=f"{help_} (default: {default})")
    return parser.parse_args()


def configure_rtx4050_guardrails(vram_fraction: float, log: logging.Logger) -> Dict[str, Any]:
    """Configure CUDA Tensor Cores, VRAM ceiling, and CPU thread pools safely."""
    safe_frac = min(max(float(vram_fraction), 0.25), 0.85)
    info = {
        "cpu_cores_total": TOTAL_CPU_CORES,
        "cpu_worker_threads_active": SAFE_WORKER_THREADS,
        "gpu_detected": False,
        "gpu_name": "CPU only (no CUDA-enabled PyTorch detected)",
        "vram_safety_cap_mb": round(VRAM_TOTAL_RTX4050_MB * safe_frac, 1),
    }
    if TORCH_AVAILABLE and torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_vram_mb = props.total_memory / (1024.0 ** 2)
        torch.cuda.set_per_process_memory_fraction(safe_frac, device=0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        info.update({
            "gpu_detected": True,
            "gpu_name": f"{dev_name} ({total_vram_mb:.0f} MB VRAM, TF32 Tensor Cores Enabled)",
            "vram_safety_cap_mb": round(total_vram_mb * safe_frac, 1),
        })
        log.info(
            "[Hardware Governor] Activated %s | VRAM Safety Cap: %.0f MB (%.0f%%) | CPU Threads: %d/%d",
            dev_name, info["vram_safety_cap_mb"], safe_frac * 100.0, SAFE_WORKER_THREADS, TOTAL_CPU_CORES,
        )
    else:
        log.info(
            "[Hardware Governor] Multi-Core CPU Mode (%d/%d threads active, 2 reserved for OS) | "
            "RTX 4050 CUDA Tensor Core hooks ready (VRAM cap set to %.0f MB)",
            SAFE_WORKER_THREADS, TOTAL_CPU_CORES, info["vram_safety_cap_mb"],
        )
    return info


def release_memory_safely() -> None:
    """Proactively free unused Python objects and CUDA cache between modules."""
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_hardware_telemetry(pinn_peak_vram_mb: float, hw_info: Dict[str, Any]) -> Dict[str, Any]:
    """Cross-platform peak GPU VRAM and system RSS RAM (works on Windows, macOS, and Linux)."""
    ram_mb = None  # never report a made-up number
    if RESOURCE_AVAILABLE:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ram_mb = ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0
    else:  # Windows: current RSS via psutil (pip install psutil)
        try:
            import psutil  # type: ignore
            ram_mb = psutil.Process().memory_info().rss / 1024.0 ** 2
        except ImportError:
            pass

    cuda_mb = 0.0
    if TORCH_AVAILABLE and torch.cuda.is_available():
        cuda_mb = torch.cuda.max_memory_allocated(0) / (1024.0 ** 2)

    vram_mb = max(cuda_mb, pinn_peak_vram_mb)
    return {
        "compute_backend": hw_info["gpu_name"],
        "cpu_threads_utilized": f"{hw_info['cpu_worker_threads_active']}/{hw_info['cpu_cores_total']} cores",
        "peak_gpu_vram_mb": round(vram_mb, 2),
        "vram_ceiling_limit_mb": hw_info["vram_safety_cap_mb"],
        "vram_constraint_satisfied": vram_mb < hw_info["vram_safety_cap_mb"],
        "peak_system_ram_mb": round(ram_mb, 2) if ram_mb is not None else "unavailable (pip install psutil)",
        "system_ram_safe_ceiling_mb": RAM_SAFE_CEILING_MB,
        "system_ram_limit_mb": RAM_TOTAL_SYSTEM_MB,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("eDNA.Orchestrator")
    args = parse_args()
    t_start = time.time()
    os.makedirs(args.export_dir, exist_ok=True)
    data_dir = args.data_dir if (args.data_dir and os.path.isdir(args.data_dir)) else None
    export = {"export_dir": args.export_dir}

    log.info("=" * 82)
    log.info("STARTING END-TO-END PHYSICS-INFORMED & METAGENOMIC eDNA ML PIPELINE")
    hw_info = configure_rtx4050_guardrails(args.vram_fraction, log)
    log.info(
        "Config: epochs_pinn=%d | samples=%d | asvs=%d | collocation_batch=%d | data_dir=%s",
        args.epochs_pinn, args.samples, args.asvs, args.batch_size, data_dir or "built-in-calibrated",
    )
    log.info("=" * 82)

    # Module 1: SIREN PINN for inverse fluid transport
    cfg = PhysicalTransportConfig(
        collocation_batch_size=args.batch_size,
        vram_safety_fraction=args.vram_fraction,
    )
    field_df = load_real_data(data_dir=data_dir, num_samples=args.samples, config=cfg, seed=args.seed)
    pinn = train_pinn_transport_model(
        field_df=field_df, config=cfg, epochs=args.epochs_pinn,
        collocation_batch_size=args.batch_size, seed=args.seed,
    )
    release_memory_safely()

    # Module 2: seqidist QC & read-length decay kinetics
    qc = run_bioinformatics_qc_pipeline(
        field_df=field_df, pinn_results=pinn, data_dir=data_dir,
        identity_threshold=0.92, seed=args.seed, **export,
    )
    release_memory_safely()

    # Module 3: low-rank deep JSDM + explainable AI
    jsdm = run_deep_jsdm_pipeline(
        field_df=field_df, qc_results=qc, pinn_results=pinn, data_dir=data_dir,
        latent_rank=4, epochs=args.jsdm_epochs, seed=args.seed, **export,
    )
    release_memory_safely()

    # Module 4: taxonomy-free ML for dark diversity & biotic indices
    tax = run_taxonomy_free_ml_pipeline(
        field_df=field_df, env_df=jsdm["env_df"], num_asvs=args.asvs,
        data_dir=data_dir, n_folds=5, seed=args.seed, **export,
    )
    release_memory_safely()

    # Module 5: ONNX export & dashboards
    onnx = export_open_edna_onnx(pinn_results=pinn, filename="open_edna_model.onnx", **export)
    viz_inputs = dict(
        field_df=field_df, pinn_results=pinn, qc_results=qc,
        jsdm_results=jsdm, tax_free_results=tax, **export,
    )
    png = generate_visual_dashboard(filename="edna_pipeline_visual_dashboard.png", **viz_inputs)
    html = generate_interactive_html_dashboard(
        onnx_info=onnx, png_dashboard_path=png,
        filename="edna_interactive_presentation.html", **viz_inputs,
    )
    release_memory_safely()

    elapsed = round(time.time() - t_start, 2)
    cp = pinn["converged_params"]
    telemetry = get_hardware_telemetry(cp["peak_vram_mb"], hw_info)

    summary = {
        "execution_time_seconds": elapsed,
        "hardware_telemetry": telemetry,
        "module_1_pinn_inverse_transport": cp,
        "module_2_bioinformatics_qc": {
            k: qc[k] for k in ("class_counts", "false_positives_filtered")
        },
        "module_3_deep_jsdm_xai": {
            "holdout_mae": jsdm["baseline_mae"],
            "holdout_richness_r2": jsdm["richness_r2"],
            "train_richness_r2": jsdm["train_richness_r2"],
            "ranked_drivers": jsdm["perm_importance"],
        },
        "module_4_taxonomy_free_ml": {
            k: tax[k] for k in ("cv_r2", "cv_mse", "num_asvs", "sparsity_pct", "fold_metrics")
        },
        "module_5_deliverables": {
            "onnx_model": onnx,
            "visual_dashboard_png": png,
            "interactive_html_dashboard": html,
            "csv_datasets": [
                qc["qc_csv_path"], qc["pa_csv_path"],
                jsdm["env_csv_path"], jsdm["corrected_pa_csv_path"], tax["asv_csv_path"],
            ],
        },
    }
    with open(os.path.join(args.export_dir, "pipeline_run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    top = jsdm["perm_importance"][0]
    log.info("=" * 82)
    log.info("PIPELINE COMPLETED SUCCESSFULLY IN %.2f SECONDS", elapsed)
    log.info(
        "  • Module 1 (SIREN PINN)   : u=%.3f m/s (true %.2f), D=%.2f m^2/s (true %.2f), "
        "k=%.5f s^-1 (true %.5f), S_x=%.2f km (true %.2f)",
        cp["u_m_s"], cp["true_u_m_s"], cp["D_m2_s"], cp["true_D_m2_s"],
        cp["k_s_inv"], cp["true_k_s_inv"], cp["S_x_km"], cp["true_Sx_km"],
    )
    log.info("  • Module 2 (seqidist QC)  : %s | Filtered %d false positives",
             qc["class_counts"], qc["false_positives_filtered"])
    log.info("  • Module 3 (Deep JSDM)    : Hold-out richness R^2=%.4f | Top Driver: %s (Delta MAE=+%.4f)",
             jsdm["richness_r2"], top["label"], top["delta_mae"])
    log.info("  • Module 4 (Tax-Free ML)  : 5-Fold CV R^2=%.4f, MSE=%.5f across %d ASVs",
             tax["cv_r2"], tax["cv_mse"], tax["num_asvs"])
    log.info(
        "  • Module 5 (RTX 4050 Safe): %s (%.2f KB) | Peak VRAM=%.2f MB (Cap: %.0f MB) | System RAM=%s MB / 16384 MB",
        os.path.basename(onnx["onnx_path"]), onnx["size_kb"],
        telemetry["peak_gpu_vram_mb"], telemetry["vram_ceiling_limit_mb"], telemetry["peak_system_ram_mb"],
    )
    log.info("=" * 82)


if __name__ == "__main__":
    main()
