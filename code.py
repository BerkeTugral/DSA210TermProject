import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

sales = pd.read_csv("vgsales.csv")
meta  = pd.read_csv("games-data 2.csv")

print("Sales shape:", sales.shape)
print("Metacritic shape:", meta.shape)

sales = sales[["Name", "Platform", "Year", "Genre", "Publisher",
               "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]]

meta = meta.rename(columns={
    "name": "Name",
    "platform": "Platform",
    "r-date": "Release_Date",
    "score": "Critic_Score",
    "user score": "User_Score"
})

keep_cols_meta = ["Name", "Platform", "Release_Date", "Critic_Score", "User_Score"]
meta = meta[[c for c in keep_cols_meta if c in meta.columns]]

sales["Name_clean"]     = sales["Name"].str.lower().str.strip()
sales["Platform_clean"] = sales["Platform"].str.lower().str.strip()
meta["Name_clean"]      = meta["Name"].str.lower().str.strip()
meta["Platform_clean"]  = meta["Platform"].str.lower().str.strip()

sales["Year"] = pd.to_numeric(sales["Year"], errors="coerce")

if "Release_Date" in meta.columns:
    meta["Year_meta"] = pd.to_datetime(meta["Release_Date"], errors="coerce").dt.year
else:
    meta["Year_meta"] = np.nan

meta["Critic_Score"] = pd.to_numeric(meta["Critic_Score"], errors="coerce")
meta["User_Score"]   = pd.to_numeric(meta["User_Score"], errors="coerce")

merged = pd.merge(
    sales,
    meta,
    on=["Name_clean", "Platform_clean"],
    how="inner",
    suffixes=("_sales", "_meta")
)

def year_close(row):
    if np.isnan(row.get("Year", np.nan)) or np.isnan(row.get("Year_meta", np.nan)):
        return True
    return abs(row["Year"] - row["Year_meta"]) <= 1

merged = merged[merged.apply(year_close, axis=1)]
merged = merged.drop(columns=["Name_clean", "Platform_clean"])

print("\nMerged shape:", merged.shape)
print("\nMerged columns:\n", merged.columns.tolist())
print("\nMerged head:\n", merged.head())

clean = merged.dropna(subset=["Global_Sales", "Critic_Score", "User_Score"]).copy()
clean = clean[clean["Global_Sales"] > 0].copy()
clean["log_sales"] = np.log1p(clean["Global_Sales"])

print("\nClean dataset shape:", clean.shape)
print("\nMissingness:\n", clean.isna().mean().sort_values(ascending=False))

plt.hist(clean["Global_Sales"], bins=50)
plt.xlabel("Global Sales (millions)")
plt.ylabel("Count")
plt.title("Distribution of Global Sales")
plt.show()

plt.hist(clean["Critic_Score"].dropna(), bins=20)
plt.xlabel("Critic Score")
plt.ylabel("Count")
plt.title("Distribution of Critic Scores")
plt.show()

plt.hist(clean["User_Score"].dropna(), bins=20)
plt.xlabel("User Score")
plt.ylabel("Count")
plt.title("Distribution of User Scores")
plt.show()

plt.hist(clean["log_sales"], bins=50)
plt.xlabel("log(1 + Global Sales)")
plt.ylabel("Count")
plt.title("Distribution of log Global Sales")
plt.show()

plt.scatter(clean["Critic_Score"], clean["Global_Sales"], alpha=0.4)
plt.xlabel("Critic Score")
plt.ylabel("Global Sales (millions)")
plt.title("Critic Score vs Global Sales")
plt.show()

plt.scatter(clean["User_Score"], clean["Global_Sales"], alpha=0.4)
plt.xlabel("User Score")
plt.ylabel("Global Sales (millions)")
plt.title("User Score vs Global Sales")
plt.show()

corr = clean[["Critic_Score", "User_Score", "Global_Sales", "log_sales"]].corr()
print("\nCorrelation matrix:\n", corr)

if "Genre" in clean.columns:
    genre_summary = clean.groupby("Genre").agg(
        mean_sales=("Global_Sales", "mean"),
        median_sales=("Global_Sales", "median"),
        mean_critic=("Critic_Score", "mean"),
        mean_user=("User_Score", "mean"),
        n=("Name_sales", "count")
    ).sort_values("mean_sales", ascending=False)
    print("\nGenre summary (top rows):\n", genre_summary.head())

r_critic, p_critic = stats.pearsonr(clean["Critic_Score"], clean["log_sales"])
r_user,   p_user   = stats.pearsonr(clean["User_Score"],   clean["log_sales"])

print("\nPearson correlation tests:")
print(f"Critic Score vs log_sales: r = {r_critic:.3f}, p = {p_critic:.3g}")
print(f"User   Score vs log_sales: r = {r_user:.3f}, p = {p_user:.3g}")

high = clean[clean["Critic_Score"] >= 80]["log_sales"]
low  = clean[clean["Critic_Score"] < 60]["log_sales"]

print("\nHigh-rated games (>=80) count:", len(high))
print("Low-rated games (<60) count:", len(low))

if len(high) > 10 and len(low) > 10:
    t_stat, p_val = stats.ttest_ind(high, low, equal_var=False)
    print("\nWelch t-test (high vs low critic score groups):")
    print(f"t = {t_stat:.3f}, p = {p_val:.3g}")
    print(f"mean log_sales (high) = {high.mean():.3f}")
    print(f"mean log_sales (low)  = {low.mean():.3f}")
else:
    print("\nNot enough data in high/low groups to run a stable t-test.")

X = clean[["Critic_Score", "User_Score"]].copy()
X = sm.add_constant(X)
y = clean["log_sales"]

model = sm.OLS(y, X).fit()
print("\nRegression summary:\n")
print(model.summary())
