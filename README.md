# eDNA transport + metagenomic ML pipeline (Brahmaputra reach demo)

Run: `python edna_pipeline.py --epochs_pinn 120 --samples 160 --asvs 250 --export_dir ./output`

| Module | File | What it does |
|---|---|---|
| 1 | `pinn_transport.py` | Simulates (or loads) eDNA concentrations along a 25 km reach, then infers transport parameters u, D, k and source (S_x, S_y). PyTorch -> SIREN + autograd PDE loss; no torch -> NumPy fit of the parameters to the analytical plume, then a SIREN surrogate is trained on the result. |
| 2 | `bioinformatics_qc.py` | Read-length mean/variance -> Fresh / Mixed / Degraded class; cosine "identity" scoring vs reference barcodes (threshold 0.92); exports QC + presence/absence CSVs. |
| 3 | `deep_jsdm.py` | Covariates (canopy, NDVI, DEM, slope, TWI) -> MLP + low-rank (L L^T) co-occurrence -> species probabilities; permutation importance. |
| 4 | `taxonomy_free_ml.py` | CLR-transformed ASV table -> regressor for Ecological Quality Status, 5-fold CV. |
| 5 | `export_and_viz.py` | Dependency-free ONNX writer, 4-panel PNG, interactive HTML dashboard. |

## Data note
The bundled `data/` files are simulated (see `data/DATA_PROVENANCE.md`).

## Changes in this version
* PINN (NumPy path): removed ground-truth leakage (a regulariser pulled u, D, k toward the *true* values).
* ONNX: exported graph now matches the model (input normalisation, all 3 SIREN layers, correct output activation). Previously the torch path silently exported identity matrices for layers 2-3 and the NumPy network was untrained (outputs ~0.03 vs 1-80 copies/mL). Verified against onnxruntime.
* JSDM: metrics/importance now computed on a hold-out split; the QC presence-absence CSV is no longer overwritten (new `transport_corrected_presence_absence.csv`).
* Taxonomy-free NumPy regressor: exact LayerNorm backward (gradient-checked), gamma/beta now trained.
* Orchestrator: works on Windows (`resource` is POSIX-only), `--jsdm_epochs` flag.

## Known limitations (not fixed)
1. **The "analytical plume" is phenomenological, not a solution of the PDE.** Its finite-difference residual in the stated PDE is about as large as the signal. The torch path therefore cannot be a faithful PINN until the forward model is replaced by a numerically solved 2-D advection-dispersion-decay PDE (or the PDE is made to match the plume). 
2. The torch code paths could not be executed in the environment used for these fixes (no space to install torch); only the NumPy paths, the ONNX writer and gradients were run and verified.
3. Synthetic targets are circular: JSDM targets come from TRUE_BETA and EQS from the simulated ASV table, so R^2 values describe the simulator, not real ecology.
4. Real-data mode only partly works: FASTQ read lengths are used, but taxon detection, embeddings and ASV tables are still simulated unless you supply `asv_counts.csv`; GeoTIFF sampling uses x_km/y_km as raster coordinates (needs a CRS transform).
