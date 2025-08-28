import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib as mpl

# ─── Style settings ────────────────────────────────────────────────────────
sns.set_style("whitegrid")
mpl.rcParams['font.family'] = 'Nimbus Roman'
mpl.rcParams['font.size'] = 24
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['axes.titlesize'] = 26
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['legend.fontsize'] = 24
mpl.rcParams['legend.title_fontsize'] = 26
mpl.rcParams['figure.titlesize'] = 28
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Nimbus Roman'
mpl.rcParams['mathtext.it'] = 'Nimbus Roman'
mpl.rcParams['mathtext.bf'] = 'Nimbus Roman'
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.03



# ─── Load and merge ────────────────────────────────────────────────────────────
df = pd.read_csv("tmid_DEB_012labels.csv")
df = df.dropna(subset=["c_label"])

# ─── Features and target ───────────────────────────────────────────────────────
# ------------------- Impute missing S_band values, if any -------------------
s_band_cols = [col for col in df.columns if col.startswith("S_ave")]

for i, col in enumerate(s_band_cols):
    if i == 0:
        next_col = s_band_cols[i+1]
        df.loc[df[col].isna(), col] = df[next_col]
    elif i == len(s_band_cols) - 1:
        prev_col = s_band_cols[i-1]
        df.loc[df[col].isna(), col] = df[prev_col]
    else:
        prev_col = s_band_cols[i-1]
        next_col = s_band_cols[i+1]
        df.loc[df[col].isna(), col] = df[[prev_col, next_col]].mean(axis=1)

ds_max_cols = [col for col in df.columns if col.startswith("dS_max")]

for i, col in enumerate(ds_max_cols):
    if i == 0:
        next_col = ds_max_cols[i+1]
        df.loc[df[col].isna(), col] = df[next_col]
    elif i == len(ds_max_cols) - 1:
        prev_col = ds_max_cols[i-1]
        df.loc[df[col].isna(), col] = df[prev_col]
    else:
        prev_col = ds_max_cols[i-1]
        next_col = ds_max_cols[i+1]
        df.loc[df[col].isna(), col] = df[[prev_col, next_col]].mean(axis=1)
# ---------------------------------------------------------------------

feature_cols = ds_max_cols  + s_band_cols #
X = df[feature_cols + ['S_bar', 'f_weighted', 'h']]
#X = df[["q2", 'q4', 'q6', 'p2', 'v', 'ent', 'q8', 'q10']]  
y = df["c_label"].astype(int)

# ─── Binarize for ROC curves ──────────────────────────────────────────────────
# classes=[0,1,2] matches Melt, Crystal, Surface
y_bin = label_binarize(y, classes=[0, 1, 2])

# ─── Train/test split ─────────────────────────────────────────────────────────
# stratify to preserve class proportions
X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
    X, y, y_bin, test_size=0.3, random_state=11, stratify=y
)

# ─── (Optional) Standard scaling ───────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)

# ─── Setup — X, y, and binarized labels ───────────────────────────────────────
X_full = df[feature_cols + ['S_bar', 'f_weighted', 'h']]
y_full = df["c_label"].astype(int)
y_bin = label_binarize(y_full, classes=[0, 1, 2])  # for RF/GB

# ─── Results storage ──────────────────────────────────────────────────────────
all_metrics = []
conf_matrices = {}

# ──────────────────────────────────────────────────────────────────────────────
# Logistic Regression (Binary: Surface vs Melt and Surface vs Crystal)
# ──────────────────────────────────────────────────────────────────────────────
lr_tasks = {
    "LR_surface_melt": [0, 2],      # Melt vs Surface
    "LR_surface_crystal": [1, 2],   # Crystal vs Surface
}


for model_name, (c0, c1) in lr_tasks.items():
    df_bin = df[df["c_label"].isin([c0, c1])].copy()
    X = df_bin[feature_cols + ['S_bar', 'f_weighted', 'h']]
    y = df_bin["c_label"].map({c0: 0, c1: 1}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=11
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=11)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    y_score = clf.predict_proba(X_test_scaled)[:, 1]

    all_metrics.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score)
    })

    conf_matrices[model_name] = confusion_matrix(y_test, y_pred)

# ──────────────────────────────────────────────────────────────────────────────
# Random Forest & Gradient Boosting (Multiclass: Surface vs Rest)
# ──────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
    X_full, y_full, y_bin, test_size=0.3, stratify=y_full, random_state=11
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

multi_models = {
    "RF": RandomForestClassifier(n_estimators=200, random_state=11),
    "GB": GradientBoostingClassifier(n_estimators=200, random_state=11),
}

class_names = ["Melt", "Crystal", "Surface"]

for model_name, model in multi_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_score = model.predict_proba(X_test_scaled)

    # Compute macro-averaged AUC
    auc_vals = [auc(*roc_curve(y_test_bin[:, i], y_score[:, i])[:2]) for i in range(3)]
    auc_macro = np.mean(auc_vals)

    all_metrics.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1-score": f1_score(y_test, y_pred, average="macro"),
        "AUC": auc_macro
    })

    conf_matrices[model_name] = confusion_matrix(y_test, y_pred)


metrics_df = pd.DataFrame(all_metrics)
metrics_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(14, 8))
sns.barplot(data=metrics_melted, x="Metric", y="Score", hue="Model", edgecolor="black", palette="Set2")
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Model Comparison — Classification")
#plt.grid(True, axis="y", alpha=0.2)
plt.legend(
    title="Model", bbox_to_anchor=(1.02, 1),
    loc="upper left", borderaxespad=0, frameon=False)
plt.tight_layout()
plt.savefig("metrics_barplot.png", bbox_inches='tight', pad_inches=0.05, dpi=300)
plt.savefig("metrics_barplot.pdf", bbox_inches='tight', pad_inches=0.05, dpi=300)
plt.show()


for model_name, cm in conf_matrices.items():
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Other", "Surface"] if "LR" in model_name else class_names
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, values_format='d')
    plt.title(f"Confusion Matrix — {model_name}")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"confmat_{model_name}.png", bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.savefig(f"confmat_{model_name}.pdf", bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.show()
