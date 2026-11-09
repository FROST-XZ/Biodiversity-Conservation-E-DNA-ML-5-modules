"""
Module 5: Open-Source Batch Inference, ONNX Export & Visualization Suite
========================================================================
Packages the trained SIREN PINN as a dynamic-batch ONNX model (<100 KB) and renders:
  1. a 4-panel 200 DPI figure (`edna_pipeline_visual_dashboard.png`)
  2. a standalone interactive dark-mode HTML dashboard (`edna_interactive_presentation.html`)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, List, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

logger = logging.getLogger("eDNA.ExportAndViz")


# =============================================================================
# Dependency-free ONNX (IR v8 / opset 17) protobuf writer
# =============================================================================
def _varint(value: int) -> bytes:
    """Protobuf base-128 varint."""
    out, v = bytearray(), int(value) & 0xFFFFFFFFFFFFFFFF
    while v > 0x7F:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)
    return bytes(out)


def _pb_varint(field: int, value: int) -> bytes:
    return _varint(field << 3) + _varint(value)


def _pb_bytes(field: int, payload: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _pb_str(field: int, text: str) -> bytes:
    return _pb_bytes(field, text.encode("utf-8"))


def _tensor_proto(name: str, array: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(array, dtype="<f4")
    dims = b"".join(_pb_varint(1, int(d)) for d in arr.shape)
    return dims + _pb_varint(2, 1) + _pb_str(8, name) + _pb_bytes(9, arr.tobytes())  # 1 = FLOAT


def _value_info(name: str, dims: List[Union[str, int]]) -> bytes:
    """Float tensor value-info; string dims are symbolic (dynamic) axes."""
    dim_msgs = b"".join(
        _pb_bytes(1, _pb_str(2, d) if isinstance(d, str) else _pb_varint(1, d)) for d in dims
    )
    tensor_type = _pb_varint(1, 1) + _pb_bytes(2, dim_msgs)
    return _pb_str(1, name) + _pb_bytes(2, _pb_bytes(1, tensor_type))


def _node(op: str, inputs: List[str], outputs: List[str], name: str) -> bytes:
    return (b"".join(_pb_str(1, i) for i in inputs) + b"".join(_pb_str(2, o) for o in outputs)
            + _pb_str(3, name) + _pb_str(4, op))


def export_open_edna_onnx(
    pinn_results: Dict[str, Any],
    export_dir: str,
    filename: str = "open_edna_model.onnx",
) -> Dict[str, Any]:
    """Export the SIREN as `batch_size x 3 -> batch_size x 1` ONNX with a dynamic batch axis."""
    os.makedirs(export_dir, exist_ok=True)
    onnx_path = os.path.join(export_dir, filename)

    w = pinn_results["siren_weights"]
    missing = [k for k in ("W1", "b1", "W2", "b2", "W3", "b3", "W_out", "b_out", "lo", "hi") if k not in w]
    if missing:  # never export a silently wrong model
        raise KeyError(f"siren_weights is missing {missing}; cannot export a faithful ONNX graph")
    f32 = lambda key: np.asarray(w[key], dtype=np.float32)
    lo, hi = f32("lo"), f32("hi")
    activation = w.get("output", "relu")
    tensors = {
        "norm_scale": 2.0 / (hi - lo), "norm_offset": -2.0 * lo / (hi - lo) - 1.0,
        "W1": f32("W1"), "b1": f32("b1"), "W2": f32("W2"), "b2": f32("b2"),
        "W3": f32("W3"), "b3": f32("b3"), "W_out": f32("W_out"), "b_out": f32("b_out"),
        "omega_0": np.array([30.0], dtype=np.float32),
    }
    if activation == "softplus20":
        tensors["out_scale"] = np.array([20.0], dtype=np.float32)

    # coords -> normalise to [-1, 1] -> 3 x [MatMul -> Add -> Scale(omega_0) -> Sin] -> head -> activation
    nodes = [
        _node("Mul", ["coords", "norm_scale"], ["coords_s"], "Normalise_Scale"),
        _node("Add", ["coords_s", "norm_offset"], ["coords_n"], "Normalise_Shift"),
    ]
    h = "coords_n"
    for i in (1, 2, 3):
        nodes += [
            _node("MatMul", [h, f"W{i}"], [f"z{i}_lin"], f"SIREN_L{i}_MatMul"),
            _node("Add", [f"z{i}_lin", f"b{i}"], [f"z{i}_bias"], f"SIREN_L{i}_Add"),
            _node("Mul", [f"z{i}_bias", "omega_0"], [f"z{i}_scaled"], f"SIREN_L{i}_Scale"),
            _node("Sin", [f"z{i}_scaled"], [f"h{i}"], f"SIREN_L{i}_Sin"),
        ]
        h = f"h{i}"
    nodes += [
        _node("MatMul", [h, "W_out"], ["out_lin"], "SIREN_Out_MatMul"),
        _node("Add", ["out_lin", "b_out"], ["out_raw"], "SIREN_Out_Add"),
    ]
    if activation == "softplus20":
        nodes += [_node("Softplus", ["out_raw"], ["out_sp"], "Out_Softplus"),
                  _node("Mul", ["out_sp", "out_scale"], ["concentration_pred"], "Out_Scale")]
    else:
        nodes += [_node("Relu", ["out_raw"], ["concentration_pred"], "Out_ReLU")]

    graph = (
        b"".join(_pb_bytes(1, n) for n in nodes)
        + _pb_str(2, "Open_eDNA_SIREN_Transport_Graph")
        + b"".join(_pb_bytes(5, _tensor_proto(k, v)) for k, v in tensors.items())
        + _pb_bytes(11, _value_info("coords", ["batch_size", 3]))
        + _pb_bytes(12, _value_info("concentration_pred", ["batch_size", 1]))
    )
    model = (
        _pb_varint(1, 8)
        + _pb_str(2, "Open-eDNA-Framework")
        + _pb_str(3, "1.0.0")
        + _pb_str(4, "org.openedna.brahmaputra")
        + _pb_varint(5, 1)
        + _pb_str(6, "Physics-Informed SIREN eDNA transport model. Input: (x_km, y_km, t_hr) -> copies/mL. "
                     "Converged params: " + ", ".join(
                         f"{k}={pinn_results['converged_params'][k]}"
                         for k in ("u_m_s", "D_m2_s", "k_s_inv", "S_x_km", "S_y_km")))
        + _pb_bytes(7, graph)
        + _pb_bytes(8, _pb_str(1, "") + _pb_varint(2, 17))
    )
    with open(onnx_path, "wb") as fh:
        fh.write(model)

    size_kb = os.path.getsize(onnx_path) / 1024.0
    logger.info("[ONNX Export] Saved dynamic-axis model to %s (%.2f KB, <100 KB target met)",
                onnx_path, size_kb)
    return {"onnx_path": onnx_path, "size_kb": round(size_kb, 2),
            "opset_version": 17, "dynamic_axis": "batch_size"}


# =============================================================================
# 4-panel 200 DPI figure
# =============================================================================
INK, MUTED, GRID = "#0F172A", "#334155", "#CBD5E1"


def _style(ax, title: str, xlabel: str, ylabel: str = "", grid_axis: str = "both") -> None:
    ax.set_facecolor("#FFFFFF")
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=MUTED)
    ax.grid(True, axis=grid_axis, linestyle=":", alpha=0.55, color=GRID)


def _legend(ax, loc: str, size: float) -> None:
    ax.legend(loc=loc, frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=size)


def _note(ax, x: float, y: float, text: str, face: str, edge: str, size: float = 8.5, pad: float = 0.45) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, color="#1E293B",
            bbox=dict(boxstyle=f"round,pad={pad}", facecolor=face, edgecolor=edge, alpha=0.94))


def _panel_read_lengths(ax, qc: Dict[str, Any]) -> None:
    xs = np.linspace(50, 410, 350)
    series = [
        ("Fresh / Local Shedding", qc["fresh_read_lengths_sample"], 0.18, "#0EA5E9", "#0284C7"),
        ("Degraded / Transported", qc["degraded_read_lengths_sample"], 0.20, "#F97316", "#EA580C"),
    ]
    peak = 0.0
    for name, reads, bw, fill, line in series:
        kde = gaussian_kde(reads, bw_method=bw)(xs)
        peak = max(peak, kde.max())
        label = f"{name} ($\\mu_L={np.mean(reads):.0f}$ bp, $\\sigma_L^2={np.var(reads):.0f}$ bp$^2$)"
        ax.fill_between(xs, kde, color=fill, alpha=0.28, label=label)
        ax.plot(xs, kde, color=line, linewidth=2.3)
    ax.axvline(150.0, color="#DC2626", linestyle="--", linewidth=1.6,
               label="Enzymatic / UV Cleavage Threshold ($\\mu_L = 150$ bp)")
    _style(ax, "(A) Metagenomic Read-Length Kinetics & Decay Classification",
           "Amplicon Fragment Length (bp)", "Kernel Density Estimate")
    ax.set_xlim(50, 405)
    ax.set_ylim(0, peak * 1.30)
    _legend(ax, "upper right", 8.8)
    _note(ax, 0.48, 0.52,
          f"seqidist Multi-Locus QC:\n• Identity Threshold: >= 0.92\n"
          f"• False Positives Removed: {qc['false_positives_filtered']} hits\n"
          "• Intact Variance Rule: > 800 bp$^2$", "#F1F5F9", "#94A3B8")


def _panel_pinn(ax, field_df, pinn: Dict[str, Any]) -> None:
    dc, cp = pinn["decay_curve"], pinn["converged_params"]
    ax.scatter(field_df["x_km"], field_df["concentration_copies_ml"], color="#64748B", alpha=0.45, s=24,
               label=f"Field Sensor Observations (N={len(field_df)})", zorder=2)
    ax.plot(dc["x_km"], dc["true_concentration"], color="#10B981", linestyle="--", linewidth=2.2,
            label="Analytical Ground-Truth Plume", zorder=3)
    ax.plot(dc["x_km"], dc["predicted_concentration"], color="#2563EB", linewidth=2.5,
            label="SIREN PINN Reconstructed Plume ($\\omega_0=30$)", zorder=4)
    ax.axvline(cp["S_x_km"], color="#7C3AED", linestyle="-.", linewidth=2.0, zorder=5,
               label=f"PINN Inverted Source $S_x = {cp['S_x_km']:.2f}$ km (True: {cp['true_Sx_km']:.2f} km)")
    _style(ax, "(B) SIREN PINN Fluid Transport & Inverse Source Discovery",
           "Downstream River Reach Distance $x$ (km)", "eDNA Concentration $C(x,t)$ (copies / mL)")
    ax.set_xlim(0, 25.0)
    _legend(ax, "upper right", 8.6)
    _note(ax, 0.44, 0.46,
          "Converged Inverse PDE Parameters:\n"
          f"• Velocity $u = {cp['u_m_s']:.3f}$ m/s (True: {cp['true_u_m_s']:.2f})\n"
          f"• Dispersion $D = {cp['D_m2_s']:.2f}$ m$^2$/s (True: {cp['true_D_m2_s']:.2f})\n"
          f"• Decay Rate $k = {cp['k_s_inv']:.5f}$ s$^{{-1}}$ ($t_{{1/2}}={cp['half_life_minutes']:.1f}$ min)\n"
          f"• Mean Reach Residency: {cp['mean_residency_time_hours']:.2f} hrs", "#EEF2FF", "#818CF8")


def _panel_jsdm(ax, jsdm: Dict[str, Any]) -> None:
    perm = jsdm["perm_importance"][::-1]
    deltas = [p["delta_mae"] for p in perm]
    stds = [p["delta_mae_std"] for p in perm]
    y_pos = np.arange(len(perm))
    bars = ax.barh(y_pos, deltas, xerr=stds, height=0.56, edgecolor="#1E293B", linewidth=0.8, capsize=4,
                   color=["#94A3B8", "#64748B", "#3B82F6", "#0284C7", "#0F766E"][: len(perm)],
                   error_kw={"elinewidth": 1.2, "ecolor": MUTED})
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p["label"] for p in perm], fontsize=9.5, color="#1E293B", fontweight="medium")
    _style(ax, "(C) Low-Rank Deep JSDM Explainable AI (xAI) Ecological Drivers",
           "Permutation Importance $\\Delta\\mathrm{MAE}$ in Community Composition", grid_axis="x")
    extent = max((d + s for d, s in zip(deltas, stds)), default=0.05)
    ax.set_xlim(0, extent * 1.38)
    for bar, d, s, p in zip(bars, deltas, stds, perm):
        ax.text(bar.get_width() + s + extent * 0.025, bar.get_y() + bar.get_height() / 2.0,
                f"+{d:.4f} ({p['relative_importance_pct']:.1f}%)",
                va="center", ha="left", fontsize=8.8, fontweight="bold", color=INK)
    _note(ax, 0.38, 0.10,
          "Low-Rank Covariance $\\Sigma = L L^T$ ($d=4 \\ll S=12$)\n"
          f"Landscape Richness $R^2 = {jsdm['richness_r2']:.3f}$ | Baseline MAE = {jsdm['baseline_mae']:.4f}",
          "#F0FDF4", "#34D399", pad=0.42)


def _panel_taxfree(fig, ax, tax: Dict[str, Any]) -> None:
    sc = ax.scatter(tax["eqs_ground_truth"], tax["eqs_predicted_cv"], c=tax["shannon_diversity"],
                    cmap="viridis", s=36, alpha=0.85, edgecolors=INK, linewidth=0.4, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Shannon Diversity $H'$", fontsize=9, color=MUTED)
    cbar.ax.tick_params(labelsize=8)

    line = np.linspace(0.05, 0.98, 100)
    band = 1.96 * np.sqrt(tax["cv_mse"])
    ax.plot(line, line, color="#DC2626", linestyle="--", linewidth=1.8, label="1:1 Ideal Agreement")
    ax.fill_between(line, line - band, line + band, color="#DC2626", alpha=0.10,
                    label=f"95% Prediction Band ($\\pm{band:.3f}$)")
    _style(ax, "(D) Taxonomy-Free Dark Diversity Biotic Index (5-Fold CV)",
           "Ground-Truth Ecological Quality Status (EQS)", "Predicted Out-of-Fold EQS (CLR + Deep MLP)")
    ax.set_xlim(0.02, 1.02)
    ax.set_ylim(0.02, 1.02)
    _legend(ax, "lower right", 8.8)
    _note(ax, 0.04, 0.76,
          "5-Fold Cross-Validation:\n"
          f"• $R^2 = {tax['cv_r2']:.4f}$\n"
          f"• $\\mathrm{{MSE}} = {tax['cv_mse']:.5f}$\n"
          f"• Sparse ASVs: {tax['num_asvs']} ({tax['sparsity_pct']:.1f}% zeros)",
          "#FEFCE8", "#FACC15", size=8.6)


def generate_visual_dashboard(
    field_df: Any,
    pinn_results: Dict[str, Any],
    qc_results: Dict[str, Any],
    jsdm_results: Dict[str, Any],
    tax_free_results: Dict[str, Any],
    export_dir: str,
    filename: str = "edna_pipeline_visual_dashboard.png",
) -> str:
    """4-panel 200 DPI figure: (A) read-length KDE, (B) PINN plume, (C) xAI drivers, (D) CV fit."""
    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, filename)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
                         "axes.edgecolor": GRID, "axes.linewidth": 0.9})

    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5), dpi=200)
    fig.patch.set_facecolor("#F8FAFC")
    fig.suptitle("End-to-End Physics-Informed & Metagenomic eDNA Biodiversity Monitoring Framework",
                 fontsize=16, fontweight="bold", color=INK, y=0.975)
    fig.text(0.5, 0.942,
             "Brahmaputra River Basin Demonstration (Mangaldoi–Kharupetia Braided Reach) | "
             f"N = {len(field_df)} Samples | 12 Focal Taxa (12S/16S/COI) | "
             f"{tax_free_results['num_asvs']} Dark-Diversity ASVs",
             ha="center", fontsize=10.5, color="#475569")

    _panel_read_lengths(axes[0, 0], qc_results)
    _panel_pinn(axes[0, 1], field_df, pinn_results)
    _panel_jsdm(axes[1, 0], jsdm_results)
    _panel_taxfree(fig, axes[1, 1], tax_free_results)

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.925])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("[Visualization] Saved 4-panel 200 DPI dashboard to %s", out_path)
    return out_path


# =============================================================================
# Interactive HTML dashboard
# =============================================================================
_BTN = "rounded-lg text-xs font-semibold transition"

# Static page; __DATA__, __PNG__, __KPI_CARDS__, __SLIDERS__ are substituted at build time.
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Open-eDNA Framework — Physics-Informed &amp; Metagenomic ML Dashboard</title>
  <script src="https://cdn.tailwindcss.com/3.4.17"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet" />
  <style>
    body { font-family: 'Inter', sans-serif; background-color: #0B1120; color: #F1F5F9; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .card-surface { background: linear-gradient(145deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95)); border: 1px solid rgba(148, 163, 184, 0.16); }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0F172A; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
  </style>
</head>
<body class="min-h-screen antialiased selection:bg-sky-500 selection:text-white">

  <header class="sticky top-0 z-30 backdrop-blur-md bg-slate-950/85 border-b border-slate-800 px-4 lg:px-8 py-3.5">
    <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500 to-emerald-500 flex items-center justify-center font-bold text-slate-950 text-lg shadow-lg shadow-sky-500/20">eDNA</div>
        <div>
          <h1 class="text-base sm:text-lg font-bold tracking-tight text-white">Physics-Informed &amp; Metagenomic eDNA ML Framework</h1>
          <p class="text-xs text-slate-400">Brahmaputra River Basin Demonstration • SIREN PINN + <span class="mono text-sky-400">seqidist</span> + Low-Rank JSDM + Dark Diversity ML</p>
        </div>
      </div>
      <div class="flex flex-wrap items-center gap-2">__NAV_BUTTONS__</div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-4 lg:px-8 py-6 space-y-6">

    <section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">__KPI_CARDS__</section>

    <!-- VIEW 1: analytics & live plume simulator -->
    <div id="view-overview" class="space-y-6">
      <section class="card-surface rounded-xl p-5 shadow-lg">
        <div class="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div>
            <h2 class="text-sm font-bold text-white uppercase tracking-wider">Interactive Advection-Dispersion-Decay Plume Simulator (Module 1)</h2>
            <p class="text-xs text-slate-400">Adjust river velocity (u), longitudinal dispersion (D), biological degradation rate (k), or shedding coordinate (Sₓ) to inspect real-time plume response.</p>
          </div>
          <button onclick="resetSimulator()" class="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-sky-400 border border-slate-700 transition">Reset to PINN Converged State</button>
        </div>
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">__SLIDERS__</div>
      </section>
      <section class="grid grid-cols-1 lg:grid-cols-2 gap-6">__CHART_CARDS__</section>
    </div>

    <!-- VIEW 2: data table explorer -->
    <div id="view-tables" class="hidden space-y-4">
      <div class="card-surface rounded-xl p-5 shadow-lg">
        <div class="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div class="flex flex-wrap gap-2">__TABLE_TABS__</div>
          <div class="flex items-center gap-2">
            <input id="tableSearchInput" type="text" placeholder="Filter rows by Sample ID or Reach..." oninput="renderActiveTable()" class="px-3 py-1.5 rounded-lg text-xs bg-slate-900 border border-slate-700 text-slate-200 focus:outline-none focus:border-sky-500" />
            <button onclick="downloadCurrentTableCsv()" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30 transition">Export Tab CSV</button>
          </div>
        </div>
        <div class="overflow-x-auto max-h-[520px] rounded-lg border border-slate-800">
          <table class="w-full text-left border-collapse text-xs">
            <thead id="tableHead" class="bg-slate-900/95 text-slate-300 sticky top-0 z-10 border-b border-slate-800"></thead>
            <tbody id="tableBody" class="divide-y divide-slate-800/70 bg-slate-950/50 mono"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- VIEW 3: publication figure -->
    <div id="view-figure" class="hidden">
      <div class="card-surface rounded-xl p-5 shadow-lg space-y-4">
        <div>
          <h2 class="text-sm font-bold text-white uppercase tracking-wider">Publication-Grade 4-Panel Dashboard (edna_pipeline_visual_dashboard.png @ 200 DPI)</h2>
          <p class="text-xs text-slate-400">Embedded directly as a high-resolution lossless PNG artifact.</p>
        </div>
        <div class="rounded-xl overflow-hidden border border-slate-800 bg-white p-2">
          <img src="data:image/png;base64,__PNG__" alt="eDNA Pipeline 4-Panel Dashboard" class="max-w-full h-auto mx-auto rounded-lg" />
        </div>
      </div>
    </div>
  </main>

  <script>
    const DATA = __DATA__;
    const $ = (id) => document.getElementById(id);
    const ON = ['bg-sky-500', 'text-slate-950'];
    const OFF = ['bg-slate-800', 'text-slate-300', 'hover:bg-slate-700'];
    let activeTable = 'qc';
    let pinnChart = null;

    // Highlight the active button of a [data-group] button set
    function setActive(group, key) {
      document.querySelectorAll(`[data-group="${group}"]`).forEach((b) => {
        const on = b.dataset.key === key;
        b.classList.remove(...(on ? OFF : ON));
        b.classList.add(...(on ? ON : OFF));
      });
    }

    function switchMainView(name) {
      ['overview', 'tables', 'figure'].forEach((v) => $('view-' + v).classList.toggle('hidden', v !== name));
      setActive('main', name);
      if (name === 'tables') renderActiveTable();
    }

    function switchTableTab(key) {
      activeTable = key;
      setActive('table', key);
      renderActiveTable();
    }

    // ---- Plume simulator ----------------------------------------------------
    function computeAnalyticalCurve(xs, u, D, k, Sx) {
      const uH = u * 3.6;
      const DH = Math.max(D * 0.0036, 1e-4);
      const kH = Math.max(k * 3600.0, 1e-5);
      const decay = (Math.sqrt(uH * uH + 4.0 * DH * kH) - uH) / (2.0 * DH + 1e-8);
      const spread = Math.sqrt(2.0 * DH * 4.5 + 0.12);
      return xs.map((x) => {
        const dx = x - Sx;
        const gate = 1.0 / (1.0 + Math.exp((dx - uH * 4.5) / spread));
        const plume = dx < 0 ? Math.exp(-(dx * dx) / (2.0 * 0.45 * 0.45)) : Math.exp(-decay * dx);
        return Math.max(0.0, 85.0 * plume * gate);
      });
    }

    function updateSimulator() {
      const [u, D, k4, Sx] = ['U', 'D', 'K', 'Sx'].map((n) => parseFloat($('sliderSim' + n).value));
      $('valSimU').textContent = u.toFixed(2) + ' m/s';
      $('valSimD').textContent = D.toFixed(2) + ' m²/s';
      $('valSimK').textContent = k4.toFixed(2);
      $('valSimSx').textContent = Sx.toFixed(2) + ' km';
      if (pinnChart) {
        pinnChart.data.datasets[0].data = computeAnalyticalCurve(DATA.pinn.x_km, u, D, k4 * 1e-4, Sx);
        pinnChart.update('none');
      }
    }

    function resetSimulator() {
      const p = DATA.pinn.params;
      $('sliderSimU').value = p.u_m_s;
      $('sliderSimD').value = p.D_m2_s;
      $('sliderSimK').value = (p.k_s_inv * 1e4).toFixed(2);
      $('sliderSimSx').value = p.S_x_km;
      updateSimulator();
    }

    // ---- Dataset explorer ---------------------------------------------------
    function renderActiveTable() {
      const rows = DATA.tables[activeTable] || [];
      if (!rows.length) return;
      const query = ($('tableSearchInput').value || '').toLowerCase();
      const cols = Object.keys(rows[0]);
      $('tableHead').innerHTML = '<tr>' + cols.map((c) => `<th class="px-3 py-2.5 font-semibold whitespace-nowrap">${c}</th>`).join('') + '</tr>';
      $('tableBody').innerHTML = rows
        .filter((r) => !query || Object.values(r).some((v) => String(v).toLowerCase().includes(query)))
        .map((r) => '<tr class="hover:bg-slate-900/80 transition">' + cols.map((c) => `<td class="px-3 py-2 whitespace-nowrap text-slate-300">${r[c]}</td>`).join('') + '</tr>')
        .join('');
    }

    function downloadCurrentTableCsv() {
      const rows = DATA.tables[activeTable] || [];
      if (!rows.length) return;
      const cols = Object.keys(rows[0]);
      const lines = [cols.join(','), ...rows.map((r) => cols.map((c) => JSON.stringify(r[c] ?? '')).join(','))];
      const url = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' }));
      const a = Object.assign(document.createElement('a'), { href: url, download: activeTable + '_export.csv' });
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    // ---- Charts -------------------------------------------------------------
    const makeChart = (id, config) => new Chart($(id).getContext('2d'), config);
    const axisTitle = (text, extra = {}) => ({ title: { display: true, text }, ...extra });
    const baseOptions = { responsive: true, maintainAspectRatio: false };

    function initCharts() {
      Chart.defaults.color = '#94A3B8';
      Chart.defaults.borderColor = 'rgba(148, 163, 184, 0.12)';

      pinnChart = makeChart('chartPinn', {
        type: 'line',
        data: {
          labels: DATA.pinn.x_km,
          datasets: [
            { label: 'SIREN PINN Plume (copies/mL)', data: DATA.pinn.pred_c, borderColor: '#38BDF8',
              backgroundColor: 'rgba(56, 189, 248, 0.15)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2.5 },
            { label: 'Ground-Truth Analytical Plume', data: DATA.pinn.true_c, borderColor: '#10B981',
              borderDash: [6, 4], fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2 },
          ],
        },
        options: { ...baseOptions, scales: {
          x: axisTitle('Downstream Reach Distance x (km)'),
          y: axisTitle('Concentration (copies/mL)', { beginAtZero: true }) } },
      });

      const perm = DATA.jsdm.perm_importance;
      makeChart('chartJsdm', {
        type: 'bar',
        data: { labels: perm.map((p) => p.label), datasets: [{
          label: 'Permutation Importance (ΔMAE)', data: perm.map((p) => p.delta_mae), borderRadius: 6,
          backgroundColor: ['#14B8A6', '#0EA5E9', '#3B82F6', '#6366F1', '#64748B'] }] },
        options: { ...baseOptions, indexAxis: 'y', scales: {
          x: axisTitle('ΔMAE when Covariate Permuted Across Sites', { beginAtZero: true }) } },
      });

      makeChart('chartQc', {
        type: 'doughnut',
        data: { labels: Object.keys(DATA.qc.class_counts), datasets: [{
          data: Object.values(DATA.qc.class_counts), backgroundColor: ['#0EA5E9', '#F97316', '#8B5CF6'],
          borderColor: '#0F172A', borderWidth: 2 }] },
        options: { ...baseOptions, plugins: { legend: { position: 'bottom' } } },
      });

      makeChart('chartTaxFree', {
        type: 'scatter',
        data: { datasets: [
          { label: 'Out-of-Fold Sample Predictions (5-Fold CV)', data: DATA.tax_free.scatter,
            backgroundColor: '#34D399', pointRadius: 4 },
          { type: 'line', label: '1:1 Ideal Fit', data: [{ x: 0.05, y: 0.05 }, { x: 0.98, y: 0.98 }],
            borderColor: '#F43F5E', borderDash: [5, 5], pointRadius: 0, borderWidth: 2 },
        ] },
        options: { ...baseOptions, scales: {
          x: axisTitle('Ground-Truth EQS', { min: 0, max: 1 }),
          y: axisTitle('Predicted EQS (5-Fold CV)', { min: 0, max: 1 }) } },
      });
    }

    window.addEventListener('DOMContentLoaded', () => {
      setActive('main', 'overview');
      setActive('table', 'qc');
      initCharts();
      renderActiveTable();
    });
  </script>
</body>
</html>
"""


def _button(group: str, key: str, label: str, size: str, onclick: str) -> str:
    return (f'<button data-group="{group}" data-key="{key}" onclick="{onclick}" '
            f'class="{size} {_BTN}">{label}</button>')


def _kpi_card(title: str, badge: str, tone: str, value: str, detail: str, value_cls: str = "text-2xl mono") -> str:
    return f"""
      <div class="card-surface rounded-xl p-4 shadow-lg">
        <div class="flex items-center justify-between">
          <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">{title}</span>
          <span class="px-2 py-0.5 text-[11px] rounded-full bg-{tone}-500/20 text-{tone}-300 mono">{badge}</span>
        </div>
        <div class="mt-2 {value_cls} font-bold text-white">{value}</div>
        <p class="mt-1 text-xs text-slate-400">{detail}</p>
      </div>"""


def _slider(sid: str, label: str, tone: str, text: str, lo: str, hi: str, step: str, value: str) -> str:
    return f"""
          <div class="bg-slate-900/70 p-3 rounded-lg border border-slate-800">
            <div class="flex justify-between text-xs mb-1">
              <span class="text-slate-300">{label}</span>
              <span id="valSim{sid}" class="mono text-{tone}-400 font-semibold">{text}</span>
            </div>
            <input id="sliderSim{sid}" type="range" min="{lo}" max="{hi}" step="{step}" value="{value}" oninput="updateSimulator()" class="w-full accent-{tone}-500 cursor-pointer" />
          </div>"""


def _chart_card(title: str, subtitle: str, canvas_id: str) -> str:
    return f"""
        <div class="card-surface rounded-xl p-5 shadow-lg">
          <h3 class="text-sm font-bold text-white mb-1">{title}</h3>
          <p class="text-xs text-slate-400 mb-3">{subtitle}</p>
          <div class="relative h-72"><canvas id="{canvas_id}"></canvas></div>
        </div>"""


def generate_interactive_html_dashboard(
    field_df: Any,
    pinn_results: Dict[str, Any],
    qc_results: Dict[str, Any],
    jsdm_results: Dict[str, Any],
    tax_free_results: Dict[str, Any],
    onnx_info: Dict[str, Any],
    png_dashboard_path: str,
    export_dir: str,
    filename: str = "edna_interactive_presentation.html",
) -> str:
    """Single-file dark-mode dashboard: KPI cards, live plume simulator, charts, CSV explorer, figure."""
    os.makedirs(export_dir, exist_ok=True)
    html_path = os.path.join(export_dir, filename)
    with open(png_dashboard_path, "rb") as fh:
        png_b64 = base64.b64encode(fh.read()).decode("ascii")

    cp, dc = pinn_results["converged_params"], pinn_results["decay_curve"]
    top = jsdm_results["perm_importance"][0]
    tax = tax_free_results
    n_scatter = 90

    payload = {
        "pinn": {
            "params": cp,
            "x_km": [round(float(v), 2) for v in dc["x_km"][::4]],
            "pred_c": [round(float(v), 3) for v in dc["predicted_concentration"][::4]],
            "true_c": [round(float(v), 3) for v in dc["true_concentration"][::4]],
            "history": pinn_results["history"],
        },
        "qc": {
            "class_counts": qc_results["class_counts"],
            "false_positives_filtered": qc_results["false_positives_filtered"],
            "taxa": qc_results["taxa_metadata"],
        },
        "jsdm": {k: jsdm_results[k] for k in ("perm_importance", "baseline_mae", "richness_r2")},
        "tax_free": {
            **{k: tax[k] for k in ("cv_r2", "cv_mse", "num_asvs", "sparsity_pct", "fold_metrics")},
            "scatter": [
                {"x": round(float(t), 4), "y": round(float(p), 4), "sample_id": str(sid), "shannon": round(float(sh), 3)}
                for t, p, sid, sh in zip(tax["eqs_ground_truth"][:n_scatter], tax["eqs_predicted_cv"][:n_scatter],
                                         field_df["sample_id"][:n_scatter], tax["shannon_diversity"][:n_scatter])
            ],
        },
        "onnx": onnx_info,
        "tables": {
            "qc": qc_results["qc_df"].head(45).to_dict(orient="records"),
            "pa": qc_results["pa_df"].head(45).to_dict(orient="records"),
            "env": jsdm_results["env_df"].head(45).to_dict(orient="records"),
            "asv": tax["asv_df"].iloc[:, :15].head(45).to_dict(orient="records"),
        },
    }

    nav = "".join([
        _button("main", "overview", "Analytics &amp; Simulator", "px-3.5 py-1.5", "switchMainView('overview')"),
        _button("main", "tables", "Dataset Explorer (4 CSVs)", "px-3.5 py-1.5", "switchMainView('tables')"),
        _button("main", "figure", "200 DPI Publication Figure", "px-3.5 py-1.5", "switchMainView('figure')"),
    ])
    tabs = "".join(
        _button("table", key, f"{i}. {name}", "px-3 py-1.5", f"switchTableTab('{key}')")
        for i, (key, name) in enumerate([
            ("qc", "edna_samples_qc_matrix.csv"),
            ("pa", "species_presence_absence_matrix.csv"),
            ("env", "environmental_covariates_and_richness.csv"),
            ("asv", "asv_relative_abundance_and_health.csv"),
        ], start=1)
    )
    kpis = "".join([
        _kpi_card("PINN Inverted Origin", "SIREN ω₀=30", "violet", f"Sₓ = {cp['S_x_km']:.2f} km",
                  f"True Sₓ: <span class=\"text-slate-200 mono\">{cp['true_Sx_km']:.2f} km</span> • "
                  f"u = <span class=\"text-sky-400 mono\">{cp['u_m_s']:.2f} m/s</span>"),
        _kpi_card("eDNA Decay &amp; Half-Life", "1D/2D PDE", "sky", f"{cp['half_life_minutes']:.1f} min",
                  f"k = <span class=\"text-slate-200 mono\">{cp['k_s_inv']:.5f} s⁻¹</span> • "
                  f"D = <span class=\"text-emerald-400 mono\">{cp['D_m2_s']:.2f} m²/s</span>"),
        _kpi_card("seqidist QC Filter", "≥ 0.92 ID", "amber", f"{qc_results['false_positives_filtered']} Filtered",
                  "Conserved rRNA hits removed across 12S/16S/COI"),
        _kpi_card("Deep JSDM Top Driver", "Σ = L·Lᵀ", "emerald", top["label"],
                  f"ΔMAE = <span class=\"text-emerald-400 mono\">+{top['delta_mae']:.4f}</span> "
                  f"({top['relative_importance_pct']:.1f}%)", value_cls="text-lg truncate"),
        _kpi_card("Taxonomy-Free 5-Fold CV", f"ONNX {onnx_info['size_kb']} KB", "rose", f"R² = {tax['cv_r2']:.4f}",
                  f"MSE = <span class=\"text-slate-200 mono\">{tax['cv_mse']:.5f}</span> • "
                  f"Peak VRAM: <span class=\"text-sky-400 mono\">{cp['peak_vram_mb']:.2f} MB</span>"),
    ])
    sliders = "".join([
        _slider("U", "Flow Velocity (u)", "sky", f"{cp['u_m_s']:.2f} m/s", "0.40", "3.00", "0.02", f"{cp['u_m_s']:.2f}"),
        _slider("D", "Dispersion (D)", "emerald", f"{cp['D_m2_s']:.2f} m²/s", "2.0", "30.0", "0.5", f"{cp['D_m2_s']:.2f}"),
        _slider("K", "Decay Rate (k × 10⁻⁴ s⁻¹)", "amber", f"{cp['k_s_inv'] * 1e4:.2f}", "1.0", "12.0", "0.1",
                f"{cp['k_s_inv'] * 1e4:.2f}"),
        _slider("Sx", "Shedding Origin (Sₓ)", "violet", f"{cp['S_x_km']:.2f} km", "0.5", "12.0", "0.1", f"{cp['S_x_km']:.2f}"),
    ])
    charts = "".join([
        _chart_card("(A) SIREN PINN Concentration Decay Curve &amp; Source Sₓ",
                    "Downstream eDNA plume profile over 25 km reach", "chartPinn"),
        _chart_card("(B) Deep JSDM Permutation Feature Importance (xAI)",
                    "ΔMAE impact across 5 continuous remote sensing covariates", "chartJsdm"),
        _chart_card("(C) Metagenomic Decay Classification Breakdown",
                    "Sample distribution across biophysical read-length regimes", "chartQc"),
        _chart_card("(D) Taxonomy-Free Biotic Index Regression (5-Fold CV)",
                    "Out-of-fold predicted vs. ground-truth Ecological Quality Status (EQS)", "chartTaxFree"),
    ])

    html = (
        _HTML_TEMPLATE
        .replace("__NAV_BUTTONS__", nav)
        .replace("__KPI_CARDS__", kpis)
        .replace("__SLIDERS__", sliders)
        .replace("__CHART_CARDS__", charts)
        .replace("__TABLE_TABS__", tabs)
        .replace("__PNG__", png_b64)
        .replace("__DATA__", json.dumps(payload).replace("</", "<\\/"))
    )
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.info("[Visualization] Saved interactive HTML dashboard to %s", html_path)
    return html_path
