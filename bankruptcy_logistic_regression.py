import textwrap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

# Load data
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

print("Data overview")
print(df.shape)
print(df["Bankrupt?"].value_counts())
print("Missing values:", df.isnull().sum().sum())

constant_cols = df.columns[df.nunique() == 1].tolist()
print("Constant columns removed:", constant_cols)
df = df.drop(columns=constant_cols)

X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]

# Check feature ranges
print("\nLargest feature maxima:")
print(X.max().sort_values().tail(5))

corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = (upper > 0.95).sum().sum()
print("Feature pairs with correlation above 0.95:", high_corr)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("\nTrain:", X_train.shape, "bankrupt:", int(y_train.sum()))
print("Test:", X_test.shape, "bankrupt:", int(y_test.sum()))

# Scale features (fit on training data only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class weights used by balanced weighting
n_train = len(y_train)
n_bankrupt = int(y_train.sum())
n_healthy = n_train - n_bankrupt
w_bankrupt = n_train / (2 * n_bankrupt)
w_healthy = n_train / (2 * n_healthy)
print(
    f"Class weights: bankrupt {w_bankrupt:.2f}, healthy {w_healthy:.2f}, "
    f"ratio {w_bankrupt / w_healthy:.1f}"
)

# Logistic regression with balanced class weights
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))
print(classification_report(y_test, y_pred, target_names=["Not Bankrupt", "Bankrupt"]))
print(confusion_matrix(y_test, y_pred))

# Comparison: model without class weights
model_unw = LogisticRegression(max_iter=1000, random_state=42)
model_unw.fit(X_train_scaled, y_train)
y_pred_unw = model_unw.predict(X_test_scaled)
print("\nUnweighted model")
print(
    classification_report(y_test, y_pred_unw, target_names=["Not Bankrupt", "Bankrupt"])
)
print(confusion_matrix(y_test, y_pred_unw))

# 5-fold cross-validation on the training set
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = cross_validate(
    pipe,
    X_train,
    y_train,
    cv=cv,
    scoring={"recall": "recall", "precision": "precision", "roc_auc": "roc_auc"},
)
print("\n5-fold cross-validation on the training set")
for k in ["recall", "precision", "roc_auc"]:
    s = res[f"test_{k}"]
    print(
        f"{k}: mean {s.mean():.3f}, sd {s.std():.3f}, "
        f"range {s.min():.3f}-{s.max():.3f}, folds {np.round(s, 3)}"
    )

# Class distribution
counts = df["Bankrupt?"].value_counts()
fig, ax = plt.subplots(figsize=(5, 4))
counts.plot(kind="bar", ax=ax, color=["lightblue", "red"])
ax.set_xticklabels(["Not bankrupt", "Bankrupt"], rotation=0)
ax.set_ylabel("Count")
ax.set_title("Class distribution")
for i, v in enumerate(counts):
    ax.text(i, v + 50, f"{v} ({v / len(df):.1%})", ha="center")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()

# Ratios that differ most between the classes
means_bankrupt = X[y == 1].mean()
means_healthy = X[y == 0].mean()
stds = X.std().replace(0, np.nan)
effect_size = (
    ((means_bankrupt - means_healthy) / stds).abs().sort_values(ascending=False)
)
top_feats = effect_size.head(6).index.tolist()
print("\nMost discriminating ratios:", top_feats)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feat in zip(axes.flat, top_feats):
    df.boxplot(column=feat, by="Bankrupt?", ax=ax, showfliers=False)
    ax.set_title("\n".join(textwrap.wrap(feat, 35)), fontsize=9)
    ax.set_xlabel("")
    ax.set_xticklabels(["Not bankrupt", "Bankrupt"])
plt.suptitle("")
plt.tight_layout()
plt.savefig("top_features_boxplots.png", dpi=150)
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Not bankrupt", "Bankrupt"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Balanced Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(
    fpr,
    tpr,
    label=f"Balanced logistic regression (AUC = {roc_auc_score(y_test, y_proba):.3f})",
)
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()

# Correlation of profitability ratios
roa_cluster = [
    "Net Income to Total Assets",
    "ROA(A) before interest and % after tax",
    "ROA(B) before interest and depreciation after tax",
    "ROA(C) before interest and depreciation before interest",
]
corr_subset = X[roa_cluster].corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    corr_subset,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    ax=ax,
    xticklabels=["Net Income/TA", "ROA(A)", "ROA(B)", "ROA(C)"],
    yticklabels=["Net Income/TA", "ROA(A)", "ROA(B)", "ROA(C)"],
)
ax.set_title("Correlation Among Profitability Ratios")
plt.tight_layout()
plt.savefig("roa_correlation_heatmap.png", dpi=150)
plt.show()
