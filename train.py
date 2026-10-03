"""Dhaka House Rent Predictor - training script.

Run:  python train.py
Needs: the Kaggle "Dhaka House Rent" CSV in this folder
       (columns: Unnamed: 0, Location, Area, Bed, Bath, Price).
Outputs: model.joblib, results.csv, charts/*.png
"""
import glob
import os
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

os.makedirs("charts", exist_ok=True)

# ---------- 1. Load ----------
csv_files = glob.glob("*.csv")
csv_files = [f for f in csv_files if f != "results.csv"]
if not csv_files:
    raise SystemExit("Ei folder e kono dataset CSV nai. Kaggle CSV ta ekhane rakho.")
CSV_FILE = csv_files[0]
print("Using file:", CSV_FILE)

df = pd.read_csv(CSV_FILE)
print("Original shape:", df.shape)
print(df.head())

# ---------- 2. Clean ----------
df = df.rename(columns={"Price": "Rent"})  # eta asole mashik bhara
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
# hubohu ek rakam row (full address shoho) bad. Location chhoto korar AGE korte hobe,
# nahole alada alada bari ke ek bhebe felbe.
n_before = len(df)
df = df.drop_duplicates()
print(f"Duplicate rows removed: {n_before - len(df)}")


def parse_rent(value):
    """'65 Thousand' -> 65000, '1.5 Lakh' -> 150000, '45,000' -> 45000"""
    s = str(value).lower().replace(",", "")
    m = re.search(r"\d+\.?\d*", s)
    if not m:
        return np.nan
    num = float(m.group())
    if "lakh" in s:
        num *= 100_000
    elif "thousand" in s:
        num *= 1_000
    return num


def parse_number(value):
    """'1,200 sqft' -> 1200"""
    m = re.search(r"\d+\.?\d*", str(value).replace(",", ""))
    return float(m.group()) if m else np.nan


def parse_location(value):
    """'Mirpur, Dhaka' style text theke area er nam ber kora"""
    parts = [p.strip() for p in str(value).split(",")]
    return parts[-2] if len(parts) >= 2 else parts[0]


df["Rent"] = df["Rent"].apply(parse_rent)
df["Area"] = df["Area"].apply(parse_number)
df["Bed"] = df["Bed"].apply(parse_number)
df["Bath"] = df["Bath"].apply(parse_number)
df["Location"] = df["Location"].apply(parse_location)

df = df.dropna()
print("After basic cleaning:", df.shape)

# ajob outlier bad (shobcheye choto/boro 1%)
for col in ["Rent", "Area"]:
    lo, hi = df[col].quantile([0.01, 0.99])
    df = df[(df[col] >= lo) & (df[col] <= hi)]
df = df[(df["Rent"] > 0) & (df["Area"] > 0)]

# kom listing wala location ke "Other" e merge (sparse column komate)
counts = df["Location"].value_counts()
rare = counts[counts < 30].index
df["Location"] = df["Location"].where(~df["Location"].isin(rare), "Other")
print("Clean shape:", df.shape, "| locations:", df["Location"].nunique())

# ---------- 3. Features ----------
X_raw = df[["Area", "Bed", "Bath", "Location"]]
y = df["Rent"]
X = pd.get_dummies(X_raw, dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 4. Compare models ----------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
}
rows, fitted = [], {}
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    rows.append({
        "Model": name,
        "MAE (Tk)": round(mean_absolute_error(y_test, p)),
        "RMSE (Tk)": round(float(np.sqrt(mean_squared_error(y_test, p)))),
        "R2": round(r2_score(y_test, p), 3),
    })
    fitted[name] = m

results = pd.DataFrame(rows)
print(results.to_string(index=False))
results.to_csv("results.csv", index=False)

best_name = results.sort_values("R2", ascending=False).iloc[0]["Model"]
best = fitted[best_name]
print("Best model:", best_name)

# ---------- 5. Charts ----------
plt.figure(figsize=(7, 4))
plt.hist(y, bins=40)
plt.title("Rent distribution")
plt.xlabel("Monthly rent (Tk)")
plt.tight_layout()
plt.savefig("charts/rent_distribution.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 4))
plt.scatter(df["Area"], y, alpha=0.2)
plt.title("Size vs Rent")
plt.xlabel("Area (sqft)")
plt.ylabel("Monthly rent (Tk)")
plt.tight_layout()
plt.savefig("charts/size_vs_rent.png", dpi=120)
plt.close()

top = df.groupby("Location")["Rent"].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 4))
top.plot(kind="bar")
plt.title("Average rent by location (top 10)")
plt.ylabel("Tk")
plt.tight_layout()
plt.savefig("charts/rent_by_location.png", dpi=120)
plt.close()

imp = pd.Series(fitted["Random Forest"].feature_importances_, index=X.columns)
imp = imp.sort_values().tail(10)
plt.figure(figsize=(7, 4))
imp.plot(kind="barh")
plt.title("Top 10 features affecting rent (Random Forest)")
plt.tight_layout()
plt.savefig("charts/feature_importance.png", dpi=120)
plt.close()

# ---------- 6. Save for the app ----------
numeric_info = {
    c: (float(X_raw[c].min()), float(X_raw[c].median()), float(X_raw[c].max()))
    for c in ["Area", "Bed", "Bath"]
}
categorical_info = {"Location": sorted(X_raw["Location"].unique().tolist())}
joblib.dump(
    {
        "model": best,
        "columns": X.columns.tolist(),
        "numeric": numeric_info,
        "categorical": categorical_info,
        "target": "Rent",
    },
    "model.joblib",
)
print("Done. model.joblib, results.csv, charts/ toiri hoyeche.")
