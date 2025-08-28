Title: Directional Entropy Bands (DEB) + UMAP (+ optional HDBSCAN)
Purpose: Reproduce DEB features and 2D embeddings used in the manuscript.

Inputs

input.csv with step, atom_id, x, y, z, S_i, S_avg, c_label
(c_label is Optional, it is calculated as each atom's crystalinity index ground truth, please see 
"https://doi.org/10.48550/arXiv.2507.17980" for further info.)


Example

python deb_pipeline.py \
  --input ./tmid_Si_Savg.csv \
  --out ./deb_out \
  --step 1 \
  --shells 0.3816 0.7633 1.1450 1.5267 1.9080 2.2900 2.6720 \
  --Sstar -5.8 --sigma_s 0.25 \
  --standardize \
  --umap_n 10 --umap_min_dist 0.0 --umap_metric euclidean \
  --random_state 11 \
  --cluster --hdb_min_cluster 50 --hdb_min_samples 10



Outputs
deb_features.csv — per-band features: S_ave_*, S_ave_*_x/y/z, dS_max_*, plus S_bar, dS_max_over_bands, f_weighted
deb_umap2d.csv — UMAP embedding (with ent/p2/q6/... if merged)
deb_umap2d.png — UMAP scatter (color by --color_by, default ent)
deb_umap2d_hdbscan.png/.pdf — if --cluster
config.json, run.log


Notes
Shell edges should be strictly increasing. The defaults match the 6 bands used in the paper.
The feature imputation step is intentionally simple for transparency; reviewers can see exactly what is done.


Citation to OP dataset:
If you are using this code or dataset, please cite our paper: 
https://doi.org/10.20944/preprints202507.2422.v1
If you merge with the OP dataset, please cite our arXiv:    
https://doi.org/10.48550/arXiv.2507.17980