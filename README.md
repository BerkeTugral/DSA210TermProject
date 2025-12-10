# DSA210 Term Project

**Mustafa Berke Tuğral – 30295**
28 Oct 2025

---

## Motivation

This project idea emerged during a discussion about*Cyberpunk 2077 by CD Projekt Red—a game that, despite arguably being one of the best ever made, suffered a disastrous launch due to bugs and incomplete features. Its initial reception was so poor that many players avoided it for years, only returning once the developers significantly improved it and heavily promoted its renewed quality. This experience made me question how strongly player ratings influence purchasing behavior.

At the same time, a counterexample came from my own experience: I once left a negative Steam review for Mount & Blade II: Bannerlord, yet continued to play it for hundreds of hours and even purchased it as gifts for others without ever changing my review. This contradiction led to the central research question of this project:

**How reliable are critic and user ratings—particularly Metacritic scores—in predicting the commercial success of a video game?**

---

## Data Source

The project uses two datasets from Kaggle:

* **Video Game Sales**
  [https://www.kaggle.com/datasets/gregorut/videogamesales/data](https://www.kaggle.com/datasets/gregorut/videogamesales/data)
* **Metacritic Video Games Data**
  [https://www.kaggle.com/datasets/brunovr/metacritic-videogames-data](https://www.kaggle.com/datasets/brunovr/metacritic-videogames-data)

The Video Game Sales dataset includes title, platform, year, publisher, genre, and regional/global sales figures. The Metacritic dataset includes critic scores, user scores, release dates, and developer information. These shared fields make merging feasible after correcting naming and formatting inconsistencies.

---

## Data Preparation

The preparation procedure consisted of:

### Step I — Data Acquisition

Both datasets were downloaded as CSV files and loaded with pandas.

### Step II — Cleaning and Merging

Several inconsistencies required preprocessing:

* Game titles were normalized (lowercased, stripped of spaces and trademarks).
* Platform names were standardized.
* Metacritic release dates were converted into year form (`Year_meta`).
* Critic and user scores were converted into numeric formats (removing “tbd” etc.).

The datasets were merged using normalized title, normalized platform, and approximate year matching (±1 year allowed). Rows missing key values such as sales or ratings were removed, and sales of zero were excluded since they cannot be meaningfully logged.

The merging code and filtering produced:

* **Merged dataset size:** 2274 rows
* **Final cleaned dataset size:** 2092 rows

This cleaned dataset was used for all subsequent analysis.

---

# Methodology

The primary aim is to examine whether critic and user scores correlate with commercial success. To evaluate this, I:

I. Conducted exploratory data analysis (EDA) to examine distributions and relationships using histograms, scatterplots, and correlation matrices.
II. Computed Pearson correlations to quantify linear associations between scores and log-transformed sales.
III. Ran group comparison tests (Welch t-tests) to compare high-rated and low-rated games.
IV. Fit a multiple regression model to test whether critic and user scores jointly predict sales.

This combination of descriptive and inferential techniques allows a structured assessment of how strongly ratings influence game sales.

---

# Code

As a disclaimer I must state that I utilized the help of ChatGPT to help me code. However, after inspecting the code thoroughly I can confidently say that in my future projects I am able to do it on my own

******************

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

******************

---

# Expected Outcomes

Before running the analysis, I expected:

I. Bad ratings significantly reduce sales.
II. Bad ratings deter undecided consumers and increase flop likelihood.
III. Large publishers with strong reputations still experience lower sales on poorly rated titles.

---

# Exploratory Data Analysis

### Dataset Overview

After all cleaning steps, the final dataset contained **2092 games** with valid values for global sales, critic score, and user score.

### Distribution of Global Sales

The histogram of raw global sales shows a severely right-skewed distribution. Most games sell under 1 million copies, while a small set of outliers reach extremely high values (20–80 million). Because of this skew, sales were transformed using log(1 + sales). The resulting distribution, while still right-skewed, is much smoother and appropriate for statistical analysis.

### Distribution of Critic Scores

The histogram of critic scores shows that most games fall between 60 and 90, with very few below 40 or above 95. This indicates that critic ratings tend to cluster around moderate to high values.

### Distribution of User Scores

User scores are centered around 6–8, but show more variation at the low end compared to critic scores. This pattern aligns with known phenomena such as review bombing or polarized community reactions.

### Scatterplots: Ratings vs Sales

Two scatterplots—**Critic Score vs Global Sales** and **User Score vs Global Sales**—reveal similar patterns:

* A dense cluster of low-sale games across all score ranges
* A slight upward trend indicating that higher-rated games tend to achieve somewhat higher sales
* A small number of extreme outliers dominating the upper-right portions of the plot

Visually, the relationship exists but is weak.

### Correlation Matrix

The correlation values (from your console output) are:

* Critic Score vs log-sales: **r = 0.2188**
* User Score vs log-sales: **r = 0.1532**
* Critic Score vs User Score: **r = 0.5745**

These results show statistically meaningful but weak associations between ratings and commercial success.

---

# Figures

Distribution of Global Sales
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/774f0820-ea65-450c-95c8-5f53c66c088f" />

Interpretation: Shows extreme right-skew in raw sales.

Distribution of log Global Sales
<img width="640" height="480" alt="DistributionLogOfGlobalSales" src="https://github.com/user-attachments/assets/255f982a-53c9-485c-9e00-7c69c682bf19" />

Interpretation: Log transformation smooths the skew and makes analysis feasible.

Distribution of Critic Scores
<img width="640" height="480" alt="DistributionOfCriticScores" src="https://github.com/user-attachments/assets/915122ca-5020-4b77-8edb-4ee836a44fbd" />

Interpretation: Critic scores cluster between 60–90.

Distribution of User Scores
<img width="640" height="480" alt="DistributionOfUserScores" src="https://github.com/user-attachments/assets/276f1024-6fb3-4def-b952-a181e7879f57" />

Interpretation: User ratings are also centered around 6–8 but with more variance.

Critic Score vs Global Sales
<img width="640" height="480" alt="CriticScoreVsGlobalSales" src="https://github.com/user-attachments/assets/2e7be53e-9040-4e7d-82ef-a40767d08e70" />

Interpretation: Weak positive trend, strong clustering, heavy outliers.

User Score vs Global Sales
<img width="640" height="480" alt="UserScoreVsGlobalSales" src="https://github.com/user-attachments/assets/d1b9c279-23c5-4d3c-a651-2b55542c1478" />

Interpretation: Similar pattern to critic scores, but even weaker correlation.

---

# Hypothesis Testing

## Hypothesis 1 — Critic Score is positively correlated with sales

**H₀:** There is no linear relationship between critic score and log-sales.
**H₁:** Critic score is positively associated with log-sales.

From Pearson correlation:

* **r = 0.219**
* **p = 4.37 × 10⁻²⁴**

**Conclusion:**
Reject H₀. Critic scores are significantly—but weakly—associated with higher sales.

---

## Hypothesis 2 — High-rated games sell more than low-rated games

Groups:

* High-rated: Critic Score ≥ 80 → 547 games
* Low-rated: Critic Score < 60 → 408 games

Welch t-test results:

* **t = 9.286**
* **p = 1.32 × 10⁻¹⁹**
* Mean log-sales (high) = **0.512**
* Mean log-sales (low) = **0.229**

**Conclusion:**
The difference is extremely significant. High-rated games substantially outperform low-rated games in sales on average.

---

## Hypothesis 3 — Critic and user scores together predict sales

Regression model:

```
log_sales ~ Critic_Score + User_Score
```

Results:

* **R² = 0.049**
* Critic Score: significant predictor (p < 0.001)
* User Score: *not* significant (p = 0.115)

**Conclusion:**
Critic scores have predictive power, but overall, ratings explain only ~5% of sales variation. This confirms that while ratings matter, many other factors influence commercial success.

---

# Final Conclusion

This project demonstrates that:

* Ratings do influence sales, but only weakly.
* High-rated games sell significantly more than low-rated games.
* Critic ratings are more predictive than user ratings.
* Even combined, scores explain only a small fraction of commercial performance.

In short:

Reviews matter, but far less than commonly assumed.
External factors such as marketing, brand strength, platform, genre, and release timing likely have much greater impacts.
