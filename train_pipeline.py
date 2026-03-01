#!/usr/bin/env python3
"""
GenomicsGPT ML Training Pipeline
=================================
Trains XGBoost + LightGBM ensemble on ClinVar pathogenicity data.
Generates SHAP explainability analysis and evaluation plots.

Run: python train_pipeline.py
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/raw/clinvar_labeled.tsv")
MODEL_DIR = Path("data/models")
PLOT_DIR = Path("data/models/plots")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# VARIANT TYPE & CONSEQUENCE DEFINITIONS
# ═══════════════════════════════════════════════════════════

VARIANT_TYPES = [
    "single nucleotide variant", "Deletion", "Duplication", "Insertion",
    "Indel", "Microsatellite", "Inversion", "Translocation", "Complex",
]

CONSEQUENCE_PATTERNS = {
    "missense": r"p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$",
    "nonsense": r"p\.\w+Ter|p\.\w+\*",
    "frameshift": r"p\.\w+fs",
    "synonymous": r"p\.\w+=|p\.[A-Z][a-z]{2}\d+=$",
    "splice": r"c\.\d+[\+\-][12][ACGT]",
    "start_lost": r"p\.Met1",
    "inframe_del": r"p\.\w+del$",
    "inframe_ins": r"p\.\w+ins",
    "intronic": r"c\.\d+[\+\-]\d+",
    "utr": r"c\.\-\d+|c\.\*\d+",
}


# ═══════════════════════════════════════════════════════════
# 1. LOAD & LABEL DATA
# ═══════════════════════════════════════════════════════════

def load_and_label():
    print("\n  [1/6] Loading pre-filtered ClinVar data...")
    cols = [
        "#AlleleID", "Type", "Name", "GeneID", "GeneSymbol",
        "ClinicalSignificance", "ClinSigSimple", "Assembly",
        "Chromosome", "Start", "Stop",
        "ReferenceAllele", "AlternateAllele",
        "ReviewStatus", "NumberSubmitters", "VariationID",
        "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF",
    ]
    df = pd.read_csv(DATA_PATH, sep="\t", usecols=cols,
                      dtype={"Chromosome": str, "Start": "Int64", "Stop": "Int64"},
                      low_memory=False)
    print(f"         Loaded {len(df):,} rows")

    sig_map = {
        "Pathogenic": 1, "Likely pathogenic": 1,
        "Pathogenic/Likely pathogenic": 1, "Pathogenic, low penetrance": 1,
        "Benign": 0, "Likely benign": 0, "Benign/Likely benign": 0,
    }
    df = df[df["ClinicalSignificance"].isin(sig_map.keys())].copy()
    df["label"] = df["ClinicalSignificance"].map(sig_map)

    n_path = (df["label"] == 1).sum()
    n_benign = (df["label"] == 0).sum()
    print(f"         Pathogenic: {n_path:,} ({n_path/len(df):.1%})")
    print(f"         Benign:     {n_benign:,} ({n_benign/len(df):.1%})")
    return df


# ═══════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

def extract_features(df):
    print("\n  [2/6] Extracting features...")
    features = pd.DataFrame(index=df.index)
    name_col = df["Name"].fillna("")

    # Variant type one-hot
    for vt in VARIANT_TYPES:
        features[f"type_{vt.lower().replace(' ', '_')}"] = (df["Type"] == vt).astype(int)

    # Molecular consequence from HGVS
    for cname, pattern in CONSEQUENCE_PATTERNS.items():
        features[f"cons_{cname}"] = name_col.str.contains(pattern, regex=True, na=False).astype(int)

    # Loss-of-function aggregate
    features["is_lof"] = (
        features.get("cons_nonsense", 0) | features.get("cons_frameshift", 0) |
        features.get("cons_splice", 0) | features.get("cons_start_lost", 0)
    ).astype(int)

    # Allele lengths
    ref = df["ReferenceAlleleVCF"].fillna("").astype(str)
    alt = df["AlternateAlleleVCF"].fillna("").astype(str)
    features["ref_len"] = ref.str.len()
    features["alt_len"] = alt.str.len()
    features["len_diff"] = features["alt_len"] - features["ref_len"]
    features["is_snv"] = ((features["ref_len"] == 1) & (features["alt_len"] == 1)).astype(int)
    features["is_indel"] = (features["len_diff"] != 0).astype(int)

    # Position
    features["position"] = df["Start"].fillna(0).astype(float)
    features["variant_span"] = (df["Stop"].fillna(0) - df["Start"].fillna(0)).clip(lower=0).astype(float)

    # Chromosome
    chrom = df["Chromosome"].fillna("0").astype(str)
    features["chrom_autosome"] = chrom.str.match(r"^\d+$").astype(int)
    features["chrom_x"] = (chrom == "X").astype(int)
    features["chrom_y"] = (chrom == "Y").astype(int)
    features["chrom_mt"] = (chrom == "MT").astype(int)
    features["chrom_num"] = pd.to_numeric(chrom, errors="coerce").fillna(0).astype(int)

    # Review quality
    features["num_submitters"] = df["NumberSubmitters"].fillna(0).astype(int)
    review_map = {
        "practice guideline": 4, "reviewed by expert panel": 3,
        "criteria provided, multiple submitters, no conflicts": 2,
        "criteria provided, conflicting classifications": 1,
        "criteria provided, single submitter": 1,
        "no assertion criteria provided": 0, "no assertion provided": 0,
    }
    features["review_stars"] = df["ReviewStatus"].fillna("").map(review_map).fillna(0).astype(int)

    # Gene-level features
    gene_stats = df.groupby("GeneSymbol")["label"].agg(["sum", "count"]).reset_index()
    gene_stats.columns = ["GeneSymbol", "gene_path_count", "gene_total_variants"]
    gene_stats["gene_path_ratio"] = gene_stats["gene_path_count"] / gene_stats["gene_total_variants"].clip(lower=1)
    gmap = gene_stats.set_index("GeneSymbol").to_dict()
    gene_col = df["GeneSymbol"].fillna("")
    features["gene_path_count"] = gene_col.map(gmap.get("gene_path_count", {})).fillna(0).astype(int)
    features["gene_total_variants"] = gene_col.map(gmap.get("gene_total_variants", {})).fillna(0).astype(int)
    features["gene_path_ratio"] = gene_col.map(gmap.get("gene_path_ratio", {})).fillna(0).astype(float)

    # Name complexity
    features["name_length"] = name_col.str.len()
    features["has_protein_change"] = name_col.str.contains(r"p\.", regex=True, na=False).astype(int)
    features["has_cdna_change"] = name_col.str.contains(r"c\.", regex=True, na=False).astype(int)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"         {features.shape[1]} features extracted")
    return features


# ═══════════════════════════════════════════════════════════
# 3. EVALUATE
# ═══════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, y_prob, name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    m = {
        "name": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "pr_precision": prec.tolist(), "pr_recall": rec.tolist(),
    }
    print(f"    {name}: AUC={m['roc_auc']:.4f}  Acc={m['accuracy']:.4f}  "
          f"F1={m['f1_macro']:.4f}  Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}")
    return m


# ═══════════════════════════════════════════════════════════
# 4. TRAIN
# ═══════════════════════════════════════════════════════════

def train_models(X_train, y_train, X_test, y_test):
    import xgboost as xgb
    import lightgbm as lgb

    print("\n  [4/6] Training models...")

    # XGBoost
    print("    Training XGBoost (300 trees, depth=6)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=len(y_train[y_train==0])/max(len(y_train[y_train==1]),1),
        random_state=42, eval_metric="logloss", early_stopping_rounds=20, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_model.predict(X_test)
    xgb_metrics = evaluate(y_test, xgb_pred, xgb_prob, "XGBoost")

    # LightGBM
    print("    Training LightGBM (300 trees, depth=6)...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, is_unbalance=True,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
    lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = lgb_model.predict(X_test)
    lgb_metrics = evaluate(y_test, lgb_pred, lgb_prob, "LightGBM")

    # Ensemble
    ens_prob = (xgb_prob + lgb_prob) / 2
    ens_pred = (ens_prob >= 0.5).astype(int)
    ens_metrics = evaluate(y_test, ens_pred, ens_prob, "Ensemble")

    return {
        "xgb_model": xgb_model, "lgb_model": lgb_model,
        "xgb_metrics": xgb_metrics, "lgb_metrics": lgb_metrics, "ens_metrics": ens_metrics,
        "xgb_prob": xgb_prob, "lgb_prob": lgb_prob, "ens_prob": ens_prob,
        "ens_pred": ens_pred,
    }


# ═══════════════════════════════════════════════════════════
# 5. SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════

def run_shap(model, X_test, feature_names):
    import shap
    print("\n  [5/6] Computing SHAP values (5000 sample)...")
    X_sample = X_test.sample(min(5000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)

    print("\n    Top 15 Features by SHAP Importance:")
    for _, row in importance.head(15).iterrows():
        bar = "█" * int(row["mean_abs_shap"] * 30 / importance["mean_abs_shap"].max())
        print(f"      {row['feature']:30s}  {row['mean_abs_shap']:.4f}  {bar}")

    return {"shap_values": shap_values, "X_sample": X_sample, "importance": importance, "explainer": explainer}


# ═══════════════════════════════════════════════════════════
# 6. GENERATE PLOTS
# ═══════════════════════════════════════════════════════════

def generate_plots(results, shap_result, y_test, feature_names):
    import shap as shap_lib
    print("\n  [6/6] Generating evaluation plots...")

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ── Plot 1: ROC Curves ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, label, color in [("xgb_metrics", "XGBoost", "#2196F3"), ("lgb_metrics", "LightGBM", "#4CAF50"), ("ens_metrics", "Ensemble", "#FF5722")]:
        m = results[key]
        ax.plot(m["fpr"], m["tpr"], color=color, lw=2, label=f'{label} (AUC={m["roc_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Pathogenicity Classification", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "roc_curves.png", dpi=150)
    plt.close()

    # ── Plot 2: Precision-Recall Curves ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, label, color in [("xgb_metrics", "XGBoost", "#2196F3"), ("lgb_metrics", "LightGBM", "#4CAF50"), ("ens_metrics", "Ensemble", "#FF5722")]:
        m = results[key]
        ax.plot(m["pr_recall"], m["pr_precision"], color=color, lw=2, label=f'{label} (AP={m["avg_precision"]:.4f})')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pr_curves.png", dpi=150)
    plt.close()

    # ── Plot 3: Confusion Matrix ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, title) in zip(axes, [("xgb_metrics", "XGBoost"), ("lgb_metrics", "LightGBM"), ("ens_metrics", "Ensemble")]):
        m = results[key]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Benign", "Pathogenic"], yticklabels=["Benign", "Pathogenic"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{title}\nAcc={m['accuracy']:.4f} | AUC={m['roc_auc']:.4f}", fontweight="bold")
    fig.suptitle("Confusion Matrices", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 4: SHAP Summary (bar) ──
    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = shap_result["importance"].head(20)
    colors = ["#FF5722" if v > top20["mean_abs_shap"].median() else "#2196F3" for v in top20["mean_abs_shap"]]
    ax.barh(range(len(top20)), top20["mean_abs_shap"].values, color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Top 20 Features — SHAP Importance", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "shap_importance.png", dpi=150)
    plt.close()

    # ── Plot 5: SHAP Beeswarm ──
    fig, ax = plt.subplots(figsize=(12, 10))
    shap_lib.summary_plot(shap_result["shap_values"], shap_result["X_sample"],
                          feature_names=feature_names, max_display=20, show=False)
    plt.title("SHAP Beeswarm — Feature Impact on Pathogenicity", fontweight="bold")
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 6: Model Comparison Bar Chart ──
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_to_plot = ["roc_auc", "accuracy", "f1_macro", "sensitivity", "specificity", "ppv"]
    metric_labels = ["AUC-ROC", "Accuracy", "F1 (Macro)", "Sensitivity", "Specificity", "PPV"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    for i, (key, label, color) in enumerate([("xgb_metrics", "XGBoost", "#2196F3"), ("lgb_metrics", "LightGBM", "#4CAF50"), ("ens_metrics", "Ensemble", "#FF5722")]):
        vals = [results[key][m] for m in metrics_to_plot]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # ── Plot 7: Score Distribution ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ens_prob = results["ens_prob"]
    ax.hist(ens_prob[y_test == 0], bins=100, alpha=0.7, color="#2196F3", label="Benign", density=True)
    ax.hist(ens_prob[y_test == 1], bins=100, alpha=0.7, color="#FF5722", label="Pathogenic", density=True)
    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold (0.5)")
    ax.set_xlabel("Ensemble Pathogenicity Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution — Benign vs Pathogenic", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "score_distribution.png", dpi=150)
    plt.close()

    print(f"    Saved 7 plots to {PLOT_DIR}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  GenomicsGPT ML Training Pipeline")
    print("  ClinVar Pathogenicity Classifier")
    print("=" * 60)

    # 1. Load
    df = load_and_label()

    # 2. Features
    X = extract_features(df)
    y = df["label"]
    feature_names = X.columns.tolist()

    # 3. Split
    print("\n  [3/6] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"         Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 4. Train
    results = train_models(X_train, y_train, X_test, y_test)

    # 5. SHAP
    shap_result = run_shap(results["xgb_model"], X_test, feature_names)

    # 6. Plots
    generate_plots(results, shap_result, y_test, feature_names)

    # Save models
    print("\n  Saving models and metrics...")
    with open(MODEL_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(results["xgb_model"], f)
    with open(MODEL_DIR / "lgb_model.pkl", "wb") as f:
        pickle.dump(results["lgb_model"], f)

    # Save metrics
    save_metrics = {}
    for key in ["xgb_metrics", "lgb_metrics", "ens_metrics"]:
        save_metrics[key] = {k: v for k, v in results[key].items()
                            if k not in ("fpr", "tpr", "pr_precision", "pr_recall")}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(save_metrics, f, indent=2)

    # Save SHAP importance
    shap_result["importance"].to_csv(MODEL_DIR / "shap_importance.csv", index=False)

    # Save feature names
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    print(f"\n  All artifacts saved to {MODEL_DIR}")

    # Final summary
    m = results["ens_metrics"]
    print("\n" + "=" * 60)
    print("  FINAL RESULTS — Ensemble (XGBoost + LightGBM)")
    print("=" * 60)
    print(f"  AUC-ROC:       {m['roc_auc']:.4f}")
    print(f"  Accuracy:      {m['accuracy']:.4f}")
    print(f"  F1 (Macro):    {m['f1_macro']:.4f}")
    print(f"  Sensitivity:   {m['sensitivity']:.4f}")
    print(f"  Specificity:   {m['specificity']:.4f}")
    print(f"  PPV:           {m['ppv']:.4f}")
    print(f"  NPV:           {m['npv']:.4f}")
    print(f"  Dataset:       {len(df):,} variants ({X.shape[1]} features)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
