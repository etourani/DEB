#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directional Entropy Bands (DEB) --> UMAP --> (optional) HDBSCAN.

Input CSV must contain:
  step, t, atom_id, mol_id, x, y, z, S_i, S_avg, c_label
  (c_label is Optional, it is calculated as each atom's crystalinity index ground truth, please see 
  "https://doi.org/10.48550/arXiv.2507.17980" for further info.)

Outputs (in --out):
  - deb_features.csv               (per-band features + S_bar (=S_avg), dS_max_over_bands, f_weighted)
  - deb_umap2d.csv                 (UMAP + S_avg + c_label)
  - deb_umap2d.png                 (colored by --color_by, default S_avg)
  - deb_umap2d_hdbscan.png/.pdf    (if --cluster)
  - config.json, run.log
"""

import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
import umap
try:
    import hdbscan
    HAS_HDB = True
except Exception:
    HAS_HDB = False

# --------------------------- CLI ---------------------------

def get_args():
    p = argparse.ArgumentParser(description="Compute Directional Entropy Bands and embed with UMAP.")
    p.add_argument("--input", required=True, help="CSV with columns step,t,atom_id,mol_id,x,y,z,S_i,S_avg,c_label (and box bounds if available).")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--step", type=int, default=1, help="Timestep to select from input CSV.")
    # Shells
    p.add_argument("--shells", type=float, nargs="+",
                   default=[0.3816, 0.7633, 1.1450, 1.5267, 1.9080, 2.2900, 2.6720],
                   help="Monotonic edges (Å) for radial shells; length = n_shells+1.")
    # f_i params
    p.add_argument("--Sstar", type=float, default=-5.8)
    p.add_argument("--sigma_s", type=float, default=0.25)
    # UMAP / standardization
    p.add_argument("--standardize", action="store_true", help="Standardize features before UMAP.")
    p.add_argument("--umap_n", type=int, default=10)
    p.add_argument("--umap_min_dist", type=float, default=0.0)
    p.add_argument("--umap_metric", type=str, default="euclidean")
    p.add_argument("--random_state", type=int, default=11)
    # Clustering
    p.add_argument("--cluster", action="store_true", help="Run HDBSCAN on UMAP embedding.")
    p.add_argument("--hdb_min_cluster", type=int, default=50)
    p.add_argument("--hdb_min_samples", type=int, default=10)
    # Plot color
    p.add_argument("--color_by", type=str, default="S_avg", help="Column to color UMAP by (e.g., S_avg or c_label).")
    return p.parse_args()

# ------------------------ Utils / IO -----------------------

def setup_logging(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    log_file = outdir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def _get_box_from_df(df: pd.DataFrame):
    """
    Prefer exact box bounds if present; else fall back to data span (warn).
    Returns (Lx, Ly, Lz) and positions wrapped into [0, L) for KDTree.
    """
    if set(["xlo","xhi","ylo","yhi","zlo","zhi"]).issubset(df.columns):
        xlo, xhi = float(df["xlo"].iloc[0]), float(df["xhi"].iloc[0])
        ylo, yhi = float(df["ylo"].iloc[0]), float(df["yhi"].iloc[0])
        zlo, zhi = float(df["zlo"].iloc[0]), float(df["zhi"].iloc[0])
        Lx, Ly, Lz = xhi - xlo, yhi - ylo, zhi - zlo
        X = df[["x","y","z"]].to_numpy(dtype=float)
        X[:,0] = (X[:,0] - xlo) % Lx
        X[:,1] = (X[:,1] - ylo) % Ly
        X[:,2] = (X[:,2] - zlo) % Lz
        return (Lx, Ly, Lz), X
    else:
        logging.warning("xlo/xhi/... not found; deriving box from data span.")
        X = df[["x","y","z"]].to_numpy(dtype=float)
        xmin,xmax = X[:,0].min(), X[:,0].max()
        ymin,ymax = X[:,1].min(), X[:,1].max()
        zmin,zmax = X[:,2].min(), X[:,2].max()
        Lx, Ly, Lz = (xmax - xmin), (ymax - ymin), (zmax - zmin)
        X[:,0] = (X[:,0] - xmin) % max(Lx, 1e-9)
        X[:,1] = (X[:,1] - ymin) % max(Ly, 1e-9)
        X[:,2] = (X[:,2] - zmin) % max(Lz, 1e-9)
        return (Lx, Ly, Lz), X

def save_umap_scatter(df_plot: pd.DataFrame, outpath: Path, color_col: str, title: str):
    plt.figure(figsize=(7.5, 6.5))
    c = df_plot[color_col] if color_col in df_plot.columns else None
    sc = plt.scatter(df_plot["umap1"], df_plot["umap2"], c=c, s=4, alpha=0.9, cmap="viridis")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(title)
    if c is not None and np.issubdtype(df_plot[color_col].dtype, np.number):
        plt.colorbar(sc, label=color_col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# ------------------ Core: DEB feature calc -----------------

def compute_deb_features(df_step: pd.DataFrame,
                         shells_edges: np.ndarray):
    """
    Compute per-atom, per-band:
      - S_ave_[r1_r2]
      - S_ave_[r1_r2]_x, _y, _z
      - dS_max_[r1_r2]
    Also returns dS_band_matrix for overall max.
    Uses S_i as the scalar entropy for central atom and neighbors.
    """
    # positions & entropy
    S_atom = df_step["S_i"].to_numpy(dtype=float)
    atom_ids = df_step["atom_id"].to_numpy()
    (Lx, Ly, Lz), X = _get_box_from_df(df_step)
    box = np.array([Lx, Ly, Lz], dtype=float)
    tree = cKDTree(X, boxsize=(Lx, Ly, Lz))

    # neighbor search up to r_max once
    r_max = float(shells_edges[-1])
    all_neighbors = tree.query_ball_point(X, r=r_max)  # list of arrays

    n_atoms = X.shape[0]
    n_bands = len(shells_edges) - 1

    out = {"atom_id": atom_ids}
    dS_band_matrix = np.full((n_bands, n_atoms), np.nan, dtype=float)

    half_box = box / 2.0
    def min_image(d):
        return (d + half_box) % box - half_box

    for i in range(n_atoms):
        xi = X[i]
        Si = S_atom[i]
        nbr_idx = np.array(all_neighbors[i], dtype=int)

        if nbr_idx.size == 0:
            continue

        nbr_idx = nbr_idx[nbr_idx != i]  # drop self
        if nbr_idx.size == 0:
            continue

        disp = X[nbr_idx] - xi
        disp = min_image(disp)
        d = np.linalg.norm(disp, axis=1)
        valid = d > 1e-12
        if not np.any(valid):
            continue

        nbr_idx = nbr_idx[valid]
        disp = disp[valid]
        d = d[valid]
        u = disp / d[:, None]
        Sj = S_atom[nbr_idx]

        band_ids = np.digitize(d, shells_edges, right=True)
        mask = (band_ids >= 1) & (band_ids <= n_bands)
        if not np.any(mask):
            continue
        band_ids = band_ids[mask] - 1
        Sj = Sj[mask]
        u = u[mask]
        d = d[mask]

        for b in range(n_bands):
            sel = (band_ids == b)
            if not np.any(sel):
                continue
            Sj_b = Sj[sel]
            u_b = u[sel]
            d_b = d[sel]

            S_ave = float(np.mean(Sj_b))
            Sx = float(np.mean(Sj_b * u_b[:,0]))
            Sy = float(np.mean(Sj_b * u_b[:,1]))
            Sz = float(np.mean(Sj_b * u_b[:,2]))
            dSmax = float(np.max((Sj_b - Si) / d_b))

            r1, r2 = shells_edges[b], shells_edges[b+1]
            tag = f"{r1:.4f}_{r2:.4f}"
            out.setdefault(f"S_ave_{tag}", [np.nan]*n_atoms)
            out.setdefault(f"S_ave_{tag}_x", [np.nan]*n_atoms)
            out.setdefault(f"S_ave_{tag}_y", [np.nan]*n_atoms)
            out.setdefault(f"S_ave_{tag}_z", [np.nan]*n_atoms)
            out.setdefault(f"dS_max_{tag}", [np.nan]*n_atoms)

            out[f"S_ave_{tag}"][i]   = S_ave
            out[f"S_ave_{tag}_x"][i] = Sx
            out[f"S_ave_{tag}_y"][i] = Sy
            out[f"S_ave_{tag}_z"][i] = Sz
            out[f"dS_max_{tag}"][i]  = dSmax
            dS_band_matrix[b, i]     = dSmax

    feats_df = pd.DataFrame(out)
    return feats_df, dS_band_matrix

# ----------------------- Main ------------------------------

def main():
    args = get_args()
    outdir = Path(args.out)
    setup_logging(outdir)
    logging.info("Starting DEB pipeline (single-file input: S_i, S_avg, c_label)")

    (outdir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # Load and select step
    df = pd.read_csv(args.input)
    df = df.loc[df["step"] == args.step].sort_values("atom_id").reset_index(drop=True)

    required = {"atom_id","x","y","z","S_i","S_avg"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Input missing required columns: {missing}")

    # Compute DEB features
    shells_edges = np.array(args.shells, dtype=float)
    if np.any(np.diff(shells_edges) <= 0):
        raise ValueError("--shells must be strictly increasing.")
    feats_df, dS_band_matrix = compute_deb_features(df, shells_edges)

    # Add S_bar (= S_avg) and f_i
    feats_df["S_bar"] = df["S_avg"].to_numpy(dtype=float)
    dS_overall = np.nanmax(dS_band_matrix, axis=0)
    feats_df["dS_max_over_bands"] = dS_overall
    feats_df["f_weighted"] = np.exp(-((feats_df["S_bar"] - args.Sstar)**2) / (2.0 * (args.sigma_s**2))) * dS_overall

    # Save features
    deb_csv = outdir / "deb_features.csv"
    feats_df.to_csv(deb_csv, index=False)
    logging.info(f"Wrote DEB features: {deb_csv}")

    # Build feature matrix for UMAP
    s_band_cols = [c for c in feats_df.columns if c.startswith("S_ave_")]
    ds_cols     = [c for c in feats_df.columns if c.startswith("dS_max_")]
    fi_cols     = ["f_weighted"]
    ent_cols    = ["S_bar"]  

    feature_cols = s_band_cols + ds_cols + fi_cols + ent_cols
    X = feats_df[feature_cols].to_numpy(dtype=float)

    # Simple NaN imputation
    def _nan_fill_by_adjacent(arr_1d):
        a = arr_1d.copy()
        if not np.any(np.isnan(a)):
            return a
        fwd = pd.Series(a).ffill().to_numpy()
        bwd = pd.Series(a).bfill().to_numpy()
        a[np.isnan(a)] = 0.5 * (fwd[np.isnan(a)] + bwd[np.isnan(a)])
        return a

    X_filled = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.any(np.isnan(col)):
            X_filled[:, j] = _nan_fill_by_adjacent(col)

    X_use = StandardScaler().fit_transform(X_filled) if args.standardize else X_filled

    # UMAP
    reducer = umap.UMAP(
        n_neighbors=args.umap_n,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        n_components=2,
        random_state=args.random_state,
    )
    emb = reducer.fit_transform(X_use)
    umap_df = pd.DataFrame({
        "atom_id": feats_df["atom_id"],
        "umap1": emb[:,0],
        "umap2": emb[:,1],
        "S_avg": df.get("S_avg", np.full(len(feats_df), np.nan)),
        "c_label": df.get("c_label", np.full(len(feats_df), np.nan)),
    })

    # Color selection
    color_col = args.color_by if args.color_by in umap_df.columns else "S_avg"
    umap_csv = outdir / "deb_umap2d.csv"
    umap_df.to_csv(umap_csv, index=False)
    save_umap_scatter(umap_df, outdir / "deb_umap2d.png", color_col, "DEB → UMAP(2D)")

    # Optional HDBSCAN
    if args.cluster:
        if not HAS_HDB:
            logging.error("HDBSCAN not installed; cannot cluster.")
        else:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=args.hdb_min_cluster,
                                        min_samples=args.hdb_min_samples)
            labels = clusterer.fit_predict(umap_df[["umap1","umap2"]].to_numpy())
            umap_df["cluster"] = labels
            umap_df.to_csv(outdir / "deb_umap2d_with_clusters.csv", index=False)

            # plot clusters
            plt.figure(figsize=(7.5, 6.5))
            uniq = sorted(set(labels))
            from matplotlib import cm
            palette = [cm.get_cmap("tab10")(i % 10) for i in range(max(1, len(uniq)))]
            cmap = {lab: (0.7,0.7,0.7,1.0) if lab == -1 else palette[k % len(palette)]
                    for k, lab in enumerate(uniq)}
            cols = [cmap[l] for l in labels]
            plt.scatter(umap_df["umap1"], umap_df["umap2"], c=cols, s=4, alpha=0.95)
            plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
            n_clu = len([u for u in uniq if u != -1])
            plt.title(f"DEB --> UMAP(2D) + HDBSCAN (n={n_clu})")
            plt.xticks([]); plt.yticks([])
            plt.tight_layout()
            plt.savefig(outdir / "deb_umap2d_hdbscan.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
            #plt.savefig(outdir / "deb_umap2d_hdbscan.pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
            plt.close()

    logging.info("Done.")

if __name__ == "__main__":
    main()
