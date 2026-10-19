from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
SEED = 42

# 1. Load data: drop the Id column, "?" = missing value, class 2 -> 0 (benign), 4 -> 1 (malignant)
df = pd.read_csv(BASE_DIR / "breast-cancer-wisconsin.csv", skipinitialspace=True, na_values="?")
df.columns = df.columns.str.strip()
df = df.drop(columns="Id")
df["Class"] = df["Class"].map({2: 0, 4: 1})
X, y = df.drop(columns="Class"), df["Class"]


def build_model(scaler):
    # Missing values -> median (learned on training data only), then scaling, then SVM
    return make_pipeline(SimpleImputer(strategy="median"), scaler, SVC(kernel="rbf"))


# 2. Compare scalers: accuracy over 30 random train/test splits
splits = StratifiedShuffleSplit(n_splits=30, test_size=0.2, random_state=SEED)
scalers = {"no scaling": "passthrough", "MinMaxScaler": MinMaxScaler(), "StandardScaler": StandardScaler()}
for name, scaler in scalers.items():
    scores = cross_val_score(build_model(scaler), X, y, cv=splits)
    print(f"{name:15s} mean accuracy = {scores.mean():.4f}  std = {scores.std():.4f}")

# 3. Final model (MinMaxScaler: all options tie, scaling is recommended): train on 80%, test on 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
model = build_model(MinMaxScaler()).fit(X_train, y_train)
print(f"\nTest accuracy: {model.score(X_test, y_test):.4f}")
print("Confusion matrix (rows = true, columns = predicted; benign, malignant):")
print(confusion_matrix(y_test, model.predict(X_test)))

# 4. Variable importance: accuracy drop when a feature's values are shuffled
imp = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=SEED)
importance = pd.Series(imp.importances_mean, index=X.columns).sort_values()
print("\nPermutation importance:")
print(importance[::-1].round(4).to_string())
importance.plot.barh(title="Permutation importance (drop in test accuracy)")
plt.tight_layout()
plt.savefig(BASE_DIR / "permutation_importance.png", dpi=150)
plt.close()

# 5. Support vectors, plotted on the two most important features
f1, f2 = importance.index[-1], importance.index[-2]
support = X_train.iloc[model[-1].support_]
print(f"\nSupport vectors: {len(support)} of {len(X_train)} training samples")
rng = np.random.default_rng(SEED)
jitter = lambda s: s + rng.uniform(-0.2, 0.2, len(s))  # features are integers, so points overlap
plt.figure(figsize=(7, 5))
for label, color, name in [(0, "tab:blue", "benign"), (1, "tab:red", "malignant")]:
    part = X_train[y_train == label]
    plt.scatter(jitter(part[f1]), jitter(part[f2]), s=14, alpha=0.4, color=color, label=name)
plt.scatter(jitter(support[f1]), jitter(support[f2]), s=70, facecolors="none", edgecolors="black", label="support vectors")
plt.xlabel(f1)
plt.ylabel(f2)
plt.title("Training data and support vectors")
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "support_vectors.png", dpi=150)
plt.close()
