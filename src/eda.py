"""
eda.py  (Task 2 - Exploratory Data Analysis)
--------------------------------------------
Reads data/tickets_clean.csv, prints summary numbers and saves charts to screenshots/.

Run (after preprocessing.py):  python src/eda.py
"""

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # lets us save charts without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "tickets_clean.csv"
OUT_DIR = ROOT / "screenshots"

OUT_DIR.mkdir(exist_ok=True)  # create screenshots/ if it is missing
df = pd.read_csv(CLEAN_PATH)

# ---------------------------------------------------------------- numbers
print("=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)
print(f"Total number of tickets: {len(df)}")
print("\nTickets per category:\n", df["category"].value_counts())
print("\nTickets per priority:\n", df["priority"].value_counts())
print("\nTickets per status:\n", df["status"].value_counts())

print("\nPercentage share of each category:")
print((df["category"].value_counts(normalize=True) * 100).round(1))

# Extra observation: how long are the descriptions?
df["word_count"] = df["clean_description"].str.split().str.len()
print("\nAverage words per description, by category:")
print(df.groupby("category")["word_count"].mean().round(1))

# Extra observation: most frequent word in each category (ignoring very common words)
STOP = {"the", "i", "to", "a", "my", "is", "and", "in", "of", "me", "for", "on",
        "it", "that", "when", "not", "this", "with", "are", "we", "an", "be", "as", "our"}
print("\nTop 5 words per category:")
for cat, group in df.groupby("category"):
    words = " ".join(group["clean_description"]).split()
    common = Counter(w for w in words if w not in STOP).most_common(5)
    print(f"  {cat:18s}", [w for w, _ in common])

# ---------------------------------------------------------------- charts
sns.set_theme(style="whitegrid")

# Chart 1: category distribution
plt.figure(figsize=(8, 5))
order = df["category"].value_counts().index
ax = sns.countplot(data=df, y="category", order=order, hue="category", palette="viridis", legend=False)
ax.set_title("Chart 1: Number of Tickets per Category")
ax.set_xlabel("Number of tickets")
ax.set_ylabel("")
for container in ax.containers:
    ax.bar_label(container, padding=3)
plt.tight_layout()
plt.savefig(OUT_DIR / "chart1_category_distribution.png", dpi=150)
plt.close()

# Chart 2: priority distribution
plt.figure(figsize=(7, 5))
p_order = ["High", "Medium", "Low", "Unknown"]
p_order = [p for p in p_order if p in df["priority"].unique()]
ax = sns.countplot(data=df, x="priority", order=p_order, hue="priority", palette="rocket", legend=False)
ax.set_title("Chart 2: Number of Tickets per Priority")
ax.set_xlabel("Priority")
ax.set_ylabel("Number of tickets")
for container in ax.containers:
    ax.bar_label(container, padding=3)
plt.tight_layout()
plt.savefig(OUT_DIR / "chart2_priority_distribution.png", dpi=150)
plt.close()

# Chart 3: priority within each category (stacked bars)
cross = pd.crosstab(df["category"], df["priority"])
cross = cross[[c for c in p_order if c in cross.columns]]
ax = cross.plot(kind="bar", stacked=True, figsize=(9, 5), colormap="rocket")
ax.set_title("Chart 3: Priority Breakdown within each Category")
ax.set_xlabel("")
ax.set_ylabel("Number of tickets")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUT_DIR / "chart3_priority_by_category.png", dpi=150)
plt.close()

# Combined picture for the report  (analysis.png)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(data=df, y="category", order=order, hue="category", palette="viridis", legend=False, ax=axes[0])
axes[0].set_title("Tickets per Category")
sns.countplot(data=df, x="priority", order=p_order, hue="priority", palette="rocket", legend=False, ax=axes[1])
axes[1].set_title("Tickets per Priority")
sns.countplot(data=df, x="status", hue="status", palette="mako", legend=False, ax=axes[2])
axes[2].set_title("Tickets per Status")
plt.tight_layout()
plt.savefig(OUT_DIR / "analysis.png", dpi=150)
plt.close()

print(f"\nCharts saved in {OUT_DIR}")
