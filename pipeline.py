 
import pandas as pd
import numpy as np
import warnings, time
warnings.filterwarnings("ignore")
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score,
)
import xgboost  as xgb
import lightgbm as lgb
import catboost as cb

np.random.seed(42)

plt.rcParams.update({
    "figure.facecolor"  : "#0d1117",
    "axes.facecolor"    : "#161b22",
    "axes.edgecolor"    : "#30363d",
    "axes.labelcolor"   : "#c9d1d9",
    "axes.titlecolor"   : "#ffffff",
    "xtick.color"       : "#8b949e",
    "ytick.color"       : "#8b949e",
    "text.color"        : "#c9d1d9",
    "grid.color"        : "#21262d",
    "grid.linewidth"    : 0.8,
    "legend.facecolor"  : "#161b22",
    "legend.edgecolor"  : "#30363d",
    "font.family"       : "monospace",
})

C = {
    "green"  : "#39d353",
    "blue"   : "#58a6ff",
    "orange" : "#f78166",
    "purple" : "#bc8cff",
    "yellow" : "#e3b341",
    "cyan"   : "#76e4f7",
    "pink"   : "#ff7eb6",
    "white"  : "#ffffff",
    "dim"    : "#484f58",
}

MODEL_COLORS = {
    "RF"  : C["green"],
    "ET"  : C["blue"],
    "HGB" : C["orange"],
    "XGB" : C["purple"],
    "LGB" : C["yellow"],
    "CAT" : C["cyan"],
}



# 1.  LOAD DATA
train = pd.read_csv("TRAIN.csv")
test  = pd.read_csv("TEST.csv")

FEAT_COLS  = [c for c in train.columns if c.startswith("F")]
X_raw      = train[FEAT_COLS]
y          = train["Class"].values
test_ids   = test["ID"].values
X_test_raw = test[FEAT_COLS]

print(f"📂 Train : {train.shape}  |  Test : {test.shape}")
print(f"   Class 0 : {(y==0).sum()}  |  Class 1 : {(y==1).sum()}")

# 2.  FEATURE ENGINEERING
def engineer(df):
    d = df.copy()
    fc = FEAT_COLS
    d["row_mean"]    = df[fc].mean(axis=1)
    d["row_std"]     = df[fc].std(axis=1)
    d["row_min"]     = df[fc].min(axis=1)
    d["row_max"]     = df[fc].max(axis=1)
    d["row_range"]   = d["row_max"] - d["row_min"]
    d["row_median"]  = df[fc].median(axis=1)
    d["row_skew"]    = df[fc].skew(axis=1)
    d["row_kurt"]    = df[fc].kurt(axis=1)
    d["row_iqr"]     = df[fc].quantile(0.75, axis=1) - df[fc].quantile(0.25, axis=1)
    d["row_l1"]      = df[fc].abs().sum(axis=1)
    d["row_l2"]      = np.sqrt((df[fc] ** 2).sum(axis=1))
    d["row_neg_cnt"] = (df[fc] < 0).sum(axis=1)
    bands = [fc[0:9], fc[9:19], fc[19:29], fc[29:38], fc[38:47]]
    for i, b in enumerate(bands):
        d[f"b{i}_mean"]  = df[b].mean(axis=1)
        d[f"b{i}_std"]   = df[b].std(axis=1)
        d[f"b{i}_range"] = df[b].max(axis=1) - df[b].min(axis=1)
    for c in fc:
        d[f"{c}_log"]  = np.log1p(df[c].abs())
    return d.fillna(0)

print(" Engineering features...")
X_eng      = engineer(X_raw)
X_test_eng = engineer(X_test_raw)
print(f"   {len(FEAT_COLS)} raw → {X_eng.shape[1]} features")

scaler    = RobustScaler()
X_sc      = scaler.fit_transform(X_eng).astype(np.float32)
X_test_sc = scaler.transform(X_test_eng).astype(np.float32)

pos_w = (y == 0).sum() / (y == 1).sum()
N_FOLDS = 10
CV = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# 3.  BASE MODELS
models = {
    "RF": RandomForestClassifier(
        n_estimators=600, max_features="sqrt",
        min_samples_leaf=2, min_samples_split=4,
        class_weight="balanced", oob_score=True,
        n_jobs=-1, random_state=42,
    ),
    "ET": ExtraTreesClassifier(
        n_estimators=600, max_features="sqrt",
        min_samples_leaf=2, class_weight="balanced",
        n_jobs=-1, random_state=43,
    ),
    "HGB": HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05,
        max_depth=6, min_samples_leaf=30,
        l2_regularization=0.5, max_bins=255,
        class_weight="balanced",
        early_stopping=True, n_iter_no_change=20,
        validation_fraction=0.1, random_state=44,
    ),
    "XGB": xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.05,
        max_depth=5, subsample=0.7, colsample_bytree=0.7,
        min_child_weight=5, gamma=0.3,
        reg_alpha=1.0, reg_lambda=5.0,
        scale_pos_weight=pos_w,
        eval_metric="logloss", early_stopping_rounds=50,
        use_label_encoder=False, n_jobs=-1,
        random_state=45, verbosity=0,
    ),
    "LGB": lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.05,
        max_depth=5, num_leaves=31,
        min_child_samples=40, subsample=0.7,
        colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
        class_weight="balanced", n_jobs=-1,
        verbose=-1, random_state=46,
    ),
    "CAT": cb.CatBoostClassifier(
        iterations=600, learning_rate=0.05,
        depth=6, l2_leaf_reg=5.0,
        auto_class_weights="Balanced",
        early_stopping_rounds=50, random_seed=47, verbose=0,
    ),
}


# 4.  FOLD VISUALIZATION HELPER
def plot_fold_dashboard(fold_idx, fold_metrics, all_folds_sofar,
                        oof_proba_fold, y_val, model_name):
    
    col = MODEL_COLORS[model_name]
    fig = plt.figure(figsize=(20, 10), facecolor="#0d1117")
    fig.suptitle(
        f"  MODEL: {model_name}  ·  FOLD {fold_idx + 1} / {N_FOLDS}  "
        f"·  AUC {fold_metrics['auc']:.5f}  ·  Acc {fold_metrics['acc']:.5f}  "
        f"·  F1 {fold_metrics['f1']:.5f}",
        fontsize=13, color=col, fontweight="bold",
        x=0.5, y=0.98, ha="center",
        path_effects=[pe.withStroke(linewidth=3, foreground="#0d1117")]
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.35,
                           top=0.92, bottom=0.07,
                           left=0.06, right=0.97)

    def style(ax, title):
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.set_title(title, fontsize=9, color="#ffffff",
                     pad=6, fontweight="bold")
        ax.grid(True, alpha=0.35)

    thresh_default = 0.5

    ax0 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_val, oof_proba_fold)
    ax0.plot([0, 1], [0, 1], "--", color=C["dim"], lw=1)
    ax0.fill_between(fpr, tpr, alpha=0.15, color=col)
    ax0.plot(fpr, tpr, color=col, lw=2,
             label=f"AUC = {fold_metrics['auc']:.4f}")
    ax0.set_xlabel("False Positive Rate", fontsize=8)
    ax0.set_ylabel("True Positive Rate",  fontsize=8)
    ax0.legend(fontsize=8, loc="lower right")
    style(ax0, "ROC Curve")

    ax1 = fig.add_subplot(gs[0, 1])
    prec, rec, _ = precision_recall_curve(y_val, oof_proba_fold)
    ap = average_precision_score(y_val, oof_proba_fold)
    ax1.fill_between(rec, prec, alpha=0.15, color=col)
    ax1.plot(rec, prec, color=col, lw=2, label=f"AP = {ap:.4f}")
    ax1.axhline(y_val.mean(), color=C["dim"], lw=1, ls="--",
                label=f"Baseline = {y_val.mean():.3f}")
    ax1.set_xlabel("Recall",    fontsize=8)
    ax1.set_ylabel("Precision", fontsize=8)
    ax1.legend(fontsize=8)
    style(ax1, "Precision-Recall Curve")

    ax2 = fig.add_subplot(gs[0, 2])
    p0 = oof_proba_fold[y_val == 0]
    p1 = oof_proba_fold[y_val == 1]
    bins = np.linspace(0, 1, 50)
    ax2.hist(p0, bins=bins, alpha=0.6, color=C["blue"],
             label="Normal (0)", density=True)
    ax2.hist(p1, bins=bins, alpha=0.6, color=C["orange"],
             label="Faulty (1)", density=True)
    ax2.axvline(thresh_default, color=C["white"], lw=1.5,
                ls="--", label=f"thresh={thresh_default}")
    ax2.set_xlabel("Predicted Probability", fontsize=8)
    ax2.set_ylabel("Density", fontsize=8)
    ax2.legend(fontsize=8)
    style(ax2, "Predicted Probability Distribution")

    ax3 = fig.add_subplot(gs[1, 0])
    preds_bin = (oof_proba_fold >= thresh_default).astype(int)
    cm = confusion_matrix(y_val, preds_bin)

    cmap = LinearSegmentedColormap.from_list("fold_cm",
                                              ["#161b22", col])
    im = ax3.imshow(cm, cmap=cmap, aspect="auto")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax3.text(j, i, f"{labels[i][j]}\n{val:,}",
                     ha="center", va="center",
                     fontsize=11, fontweight="bold",
                     color="#ffffff" if val > cm.max() * 0.4 else "#8b949e")
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Pred: Normal", "Pred: Faulty"], fontsize=8)
    ax3.set_yticklabels(["True: Normal", "True: Faulty"], fontsize=8)
    plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.04)
    style(ax3, "Confusion Matrix (threshold=0.5)")

    ax4 = fig.add_subplot(gs[1, 1])
    fold_nums = list(range(1, len(all_folds_sofar) + 1))
    aucs_so_far = [m["auc"] for m in all_folds_sofar]
    ax4.plot(fold_nums, aucs_so_far, "o-", color=col,
             lw=2, ms=6, zorder=3)
    ax4.fill_between(fold_nums, aucs_so_far,
                     min(aucs_so_far) - 0.0005,
                     alpha=0.15, color=col)
    mean_auc = np.mean(aucs_so_far)
    ax4.axhline(mean_auc, color=C["yellow"], lw=1.2, ls="--",
                label=f"mean={mean_auc:.4f}")
    ax4.set_xlabel("Fold", fontsize=8)
    ax4.set_ylabel("AUC",  fontsize=8)
    ax4.set_xticks(fold_nums)
    ax4.legend(fontsize=8)
    style(ax4, "AUC Across Folds (so far)")

    ax5 = fig.add_subplot(gs[1, 2])
    f1s_so_far  = [m["f1"]  for m in all_folds_sofar]
    accs_so_far = [m["acc"] for m in all_folds_sofar]
    ax5.plot(fold_nums, f1s_so_far,  "o-", color=C["green"],
             lw=2, ms=6, label="Macro-F1", zorder=3)
    ax5.plot(fold_nums, accs_so_far, "s-", color=C["pink"],
             lw=2, ms=6, label="Accuracy", zorder=3)
    ax5.set_xlabel("Fold", fontsize=8)
    ax5.set_ylabel("Score", fontsize=8)
    ax5.set_xticks(fold_nums)
    ax5.legend(fontsize=8)
    style(ax5, "F1 & Accuracy Across Folds (so far)")

    plt.savefig(f"fold_{model_name}_{fold_idx+1:02d}.png",
                dpi=110, bbox_inches="tight",
                facecolor="#0d1117")
    plt.show()
    print(f" Saved → fold_{model_name}_{fold_idx+1:02d}.png")

# 5.  FINAL SUMMARY DASHBOARD  (across all models)

def plot_model_summary(all_model_metrics, oof_probas):
    """
    After all models complete: side-by-side comparison, ensemble proba
    distribution, and AUC/F1 leaderboard.
    """
    fig = plt.figure(figsize=(22, 11), facecolor="#0d1117")
    fig.suptitle("  ENSEMBLE SUMMARY DASHBOARD  ·  All Models & Folds",
                 fontsize=15, color=C["white"], fontweight="bold",
                 x=0.5, y=0.99, ha="center")

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.38,
                           top=0.93, bottom=0.08,
                           left=0.06, right=0.97)

    def style(ax, title):
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.set_title(title, fontsize=10, color="#ffffff",
                     pad=8, fontweight="bold")
        ax.grid(True, alpha=0.35)

    names  = list(all_model_metrics.keys())
    colors = [MODEL_COLORS[n] for n in names]

    ax0 = fig.add_subplot(gs[0, 0])
    mean_aucs = [np.mean([m["auc"] for m in all_model_metrics[n]]) for n in names]
    bars = ax0.barh(names, mean_aucs, color=colors, alpha=0.85,
                    height=0.55, zorder=3)
    ax0.set_xlim(min(mean_aucs) - 0.002, 1.0005)
    for bar, val in zip(bars, mean_aucs):
        ax0.text(val + 0.0001, bar.get_y() + bar.get_height()/2,
                 f"{val:.5f}", va="center", fontsize=8.5,
                 color=C["white"], fontweight="bold")
    ax0.invert_yaxis()
    style(ax0, "Mean OOF AUC by Model")

    ax1 = fig.add_subplot(gs[0, 1])
    mean_f1s = [np.mean([m["f1"] for m in all_model_metrics[n]]) for n in names]
    bars1 = ax1.barh(names, mean_f1s, color=colors, alpha=0.85,
                     height=0.55, zorder=3)
    ax1.set_xlim(min(mean_f1s) - 0.002, 1.0005)
    for bar, val in zip(bars1, mean_f1s):
        ax1.text(val + 0.0001, bar.get_y() + bar.get_height()/2,
                 f"{val:.5f}", va="center", fontsize=8.5,
                 color=C["white"], fontweight="bold")
    ax1.invert_yaxis()
    style(ax1, "Mean OOF Macro-F1 by Model")

    ax2 = fig.add_subplot(gs[0, 2])
    auc_per_model = [[m["auc"] for m in all_model_metrics[n]] for n in names]
    bp = ax2.boxplot(auc_per_model, vert=True, patch_artist=True,
                     medianprops=dict(color=C["white"], lw=2),
                     whiskerprops=dict(color=C["dim"]),
                     capprops=dict(color=C["dim"]),
                     flierprops=dict(marker="o", ms=4,
                                    markerfacecolor=C["orange"]))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax2.set_xticks(range(1, len(names)+1))
    ax2.set_xticklabels(names)
    style(ax2, "AUC Distribution per Model (10 folds)")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot([0,1], [0,1], "--", color=C["dim"], lw=1)
    for name in names:
        oof  = oof_probas[name]
        fpr, tpr, _ = roc_curve(y, oof)
        auc  = roc_auc_score(y, oof)
        ax3.plot(fpr, tpr, color=MODEL_COLORS[name], lw=1.8,
                 label=f"{name} {auc:.4f}")
    ax3.set_xlabel("FPR", fontsize=9)
    ax3.set_ylabel("TPR", fontsize=9)
    ax3.legend(fontsize=8, loc="lower right")
    style(ax3, "ROC Curves — All Models")

    ax4 = fig.add_subplot(gs[1, 1])
    ensemble_oof_local = sum(oof_probas[n] for n in names) / len(names)
    bins = np.linspace(0, 1, 60)
    p0 = ensemble_oof_local[y == 0]
    p1 = ensemble_oof_local[y == 1]
    ax4.hist(p0, bins=bins, alpha=0.6, color=C["blue"],
             label="Normal (0)", density=True)
    ax4.hist(p1, bins=bins, alpha=0.6, color=C["orange"],
             label="Faulty (1)", density=True)
    ax4.set_xlabel("Ensemble Probability", fontsize=9)
    ax4.set_ylabel("Density", fontsize=9)
    ax4.legend(fontsize=8)
    style(ax4, "Ensemble Probability Distribution")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    mean_accs = [np.mean([m["acc"] for m in all_model_metrics[n]]) for n in names]
    order = np.argsort(mean_aucs)[::-1]
    rows  = []
    for rank, idx in enumerate(order):
        medal = ["🥇","🥈","🥉","  4","  5","  6"][rank]
        rows.append([
            f"{medal} {names[idx]}",
            f"{mean_aucs[idx]:.5f}",
            f"{mean_f1s[idx]:.5f}",
            f"{mean_accs[idx]:.5f}",
        ])
    col_labels = ["Model", "AUC", "Macro-F1", "Accuracy"]
    tbl = ax5.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 2.0)
    for j in range(4):
        tbl[(0, j)].set_facecolor("#21262d")
        tbl[(0, j)].set_text_props(color=C["white"], fontweight="bold")

    for i, idx in enumerate(order):
        for j in range(4):
            tbl[(i+1, j)].set_facecolor("#161b22")
            tbl[(i+1, j)].set_text_props(color=colors[idx])
    ax5.set_title("  Model Leaderboard", fontsize=10,
                  color=C["white"], fontweight="bold", pad=8)

    plt.savefig("ensemble_summary.png", dpi=110, bbox_inches="tight",
                facecolor="#0d1117")
    plt.show()
    print("Saved → ensemble_summary.png")

# 6.  OOF GENERATION WITH PER-FOLD VISUALIZATION
print("\nOut-Of-Fold generation with per-fold dashboards:\n")

oof_probas       = {}
test_probas      = {}
all_model_metrics = {}

for model_name, clf in models.items():
    print(f"\n{'═'*55}")
    print(f"  MODEL : {model_name}")
    print(f"{'═'*55}")

    oof        = np.zeros(len(y))
    tst        = np.zeros(len(test_ids))
    fold_metrics_list = []

    for fold_idx, (tr_idx, va_idx) in enumerate(CV.split(X_sc, y)):
        t_fold = time.time()
        Xtr, Xva = X_sc[tr_idx], X_sc[va_idx]
        ytr, yva = y[tr_idx],    y[va_idx]

        if model_name == "XGB":
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        elif model_name == "LGB":
            clf.fit(Xtr, ytr,
                    eval_set=[(Xva, yva)],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(-1)])
        elif model_name == "CAT":
            clf.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=0)
        else:
            clf.fit(Xtr, ytr)

        fold_proba      = clf.predict_proba(Xva)[:, 1]
        oof[va_idx]     = fold_proba
        tst            += clf.predict_proba(X_test_sc)[:, 1] / N_FOLDS


        fold_preds = (fold_proba >= 0.5).astype(int)
        fm = {
            "auc" : roc_auc_score(yva, fold_proba),
            "acc" : accuracy_score(yva, fold_preds),
            "f1"  : f1_score(yva, fold_preds, average="macro"),
        }
        fold_metrics_list.append(fm)

        elapsed = time.time() - t_fold
        print(f"   Fold {fold_idx+1:>2}/{N_FOLDS}  "
              f"AUC={fm['auc']:.5f}  "
              f"Acc={fm['acc']:.5f}  "
              f"F1={fm['f1']:.5f}  "
              f"[{elapsed:.1f}s]")

        plot_fold_dashboard(
            fold_idx        = fold_idx,
            fold_metrics    = fm,
            all_folds_sofar = fold_metrics_list,
            oof_proba_fold  = fold_proba,
            y_val           = yva,
            model_name      = model_name,
        )

    overall_auc = roc_auc_score(y, oof)
    overall_f1  = f1_score(y, (oof >= 0.5).astype(int), average="macro")
    overall_acc = accuracy_score(y, (oof >= 0.5).astype(int))
    print(f"\n  {model_name} COMPLETE")
    print(f"     Overall OOF AUC={overall_auc:.5f}  "
          f"F1={overall_f1:.5f}  Acc={overall_acc:.5f}")

    oof_probas[model_name]        = oof
    test_probas[model_name]       = tst
    all_model_metrics[model_name] = fold_metrics_list

# 7.  ENSEMBLE SUMMARY DASHBOARD
print("\nRendering ensemble summary dashboard...")
plot_model_summary(all_model_metrics, oof_probas)


# 8.  WEIGHTED ENSEMBLE + THRESHOLD SWEEP
print("\n Computing ensemble weights from OOF AUCs...")
aucs    = {n: roc_auc_score(y, p) for n, p in oof_probas.items()}
total   = sum(aucs.values())
weights = {n: a / total for n, a in aucs.items()}
print("   Weights:", {n: f"{w:.4f}" for n, w in weights.items()})

ensemble_oof  = sum(weights[n] * oof_probas[n]  for n in models)
ensemble_test = sum(weights[n] * test_probas[n] for n in models)
ens_auc = roc_auc_score(y, ensemble_oof)
print(f"   Ensemble OOF AUC : {ens_auc:.5f}")

print("\n Threshold sweep (step=0.001)...")
best_thresh, best_f1_score = 0.5, 0.0
for thresh in np.arange(0.10, 0.90, 0.001):
    f1 = f1_score(y, (ensemble_oof >= thresh).astype(int), average="macro")
    if f1 > best_f1_score:
        best_f1_score = f1
        best_thresh   = thresh

best_acc = accuracy_score(y, (ensemble_oof >= best_thresh).astype(int))
print(f"   Best threshold : {best_thresh:.3f}")
print(f"   OOF Macro-F1   : {best_f1_score:.5f}")
print(f"   OOF Accuracy   : {best_acc:.5f}")
print(f"   OOF AUC        : {ens_auc:.5f}")

print("\n   Classification Report:")
print(classification_report(y, (ensemble_oof >= best_thresh).astype(int),
                              target_names=["Normal(0)", "Faulty(1)"]))

cm = confusion_matrix(y, (ensemble_oof >= best_thresh).astype(int))
print("   Confusion Matrix:")
print(f"   TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
print(f"   FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")


# 9.  THRESHOLD VISUALIZATION
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0d1117")
threshs  = np.arange(0.10, 0.90, 0.001)
f1s, accs = [], []
for t in threshs:
    preds = (ensemble_oof >= t).astype(int)
    f1s.append(f1_score(y, preds, average="macro"))
    accs.append(accuracy_score(y, preds))

for ax in axes:
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.grid(True, alpha=0.35)

axes[0].plot(threshs, f1s,  color=C["green"], lw=2, label="Macro-F1")
axes[0].plot(threshs, accs, color=C["pink"],  lw=2, label="Accuracy")
axes[0].axvline(best_thresh, color=C["yellow"], ls="--", lw=1.5,
                label=f"Best={best_thresh:.3f}")
axes[0].set_xlabel("Threshold", color="#c9d1d9")
axes[0].set_ylabel("Score",     color="#c9d1d9")
axes[0].set_title("Threshold vs Score", color=C["white"], fontweight="bold")
axes[0].legend(fontsize=8)

p0 = ensemble_oof[y == 0]
p1 = ensemble_oof[y == 1]
bins = np.linspace(0, 1, 60)
axes[1].hist(p0, bins=bins, alpha=0.65, color=C["blue"],
             density=True, label="Normal(0)")
axes[1].hist(p1, bins=bins, alpha=0.65, color=C["orange"],
             density=True, label="Faulty(1)")
axes[1].axvline(best_thresh, color=C["yellow"], ls="--", lw=2,
                label=f"thresh={best_thresh:.3f}")
axes[1].set_xlabel("Ensemble Probability", color="#c9d1d9")
axes[1].set_ylabel("Density",              color="#c9d1d9")
axes[1].set_title("Ensemble Prob Distribution", color=C["white"], fontweight="bold")
axes[1].legend(fontsize=8)

cm_final = confusion_matrix(y, (ensemble_oof >= best_thresh).astype(int))
cmap_cm = LinearSegmentedColormap.from_list("ens_cm", ["#161b22", C["green"]])
im = axes[2].imshow(cm_final, cmap=cmap_cm)
lbls = [["TN","FP"],["FN","TP"]]
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, f"{lbls[i][j]}\n{cm_final[i,j]:,}",
                     ha="center", va="center", fontsize=12,
                     fontweight="bold", color="#ffffff")
axes[2].set_xticks([0,1])
axes[2].set_yticks([0,1])
axes[2].set_xticklabels(["Pred:Normal","Pred:Faulty"], color="#8b949e")
axes[2].set_yticklabels(["True:Normal","True:Faulty"], color="#8b949e")
axes[2].set_title(f"Final Confusion Matrix @ {best_thresh:.3f}",
                   color=C["white"], fontweight="bold")
plt.colorbar(im, ax=axes[2], fraction=0.03)

final_preds = (ensemble_test >= best_thresh).astype(int)
submission  = pd.DataFrame({"ID": test_ids, "CLASS": final_preds})
submission  = submission.set_index("ID").reindex(test_ids).reset_index()
submission.to_csv("FINAL.csv", index=False)