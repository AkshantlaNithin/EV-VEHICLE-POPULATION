import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import shapiro, chi2_contingency, ttest_ind, norm, binom, poisson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Load dataset
df_ev = pd.read_csv('Electric_Vehicle_Population_Data.csv')

# Set style
sns.set(style='whitegrid')
plt.rcParams["figure.figsize"] = (14, 6)

# Clean data
df_ev_clean = df_ev.dropna(subset=["Model Year", "Make", "Electric Vehicle Type", "Electric Range"])

# ---------------------------
# OBJECTIVE 1: Line Graph
# ---------------------------
yearly_counts = df_ev_clean["Model Year"].value_counts().sort_index()
yearly_counts.plot(marker='o')
plt.title("Line Graph: Electric Vehicle Registrations by Year")
plt.xlabel("Model Year")
plt.ylabel("Number of Vehicles")
plt.tight_layout()
plt.show()

# ---------------------------
# OBJECTIVE 2: Bar Graph
# ---------------------------
top_makes = df_ev_clean["Make"].value_counts().head(10)
sns.barplot(x=top_makes.index, y=top_makes.values, palette='viridis')
plt.title("Bar Graph: Top 10 Electric Vehicle Makes")
plt.ylabel("Number of Vehicles")
plt.xlabel("Make")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------
# OBJECTIVE 3: Scatter Plot
# ---------------------------
df_plot = df_ev_clean[(df_ev_clean["Base MSRP"] > 0) & (df_ev_clean["Electric Range"] > 0)]
sns.scatterplot(data=df_plot, x="Electric Range", y="Base MSRP", hue="Electric Vehicle Type", alpha=0.6)
plt.title("Scatter Plot: Electric Range vs Base MSRP")
plt.xlabel("Electric Range (miles)")
plt.ylabel("Base MSRP ($)")
plt.tight_layout()
plt.show()

# ---------------------------
# OBJECTIVE 4: Boxplot
# ---------------------------
sns.boxplot(data=df_ev_clean, x="Electric Vehicle Type", y="Electric Range", palette="Set2")
plt.title("Boxplot: Electric Range by Vehicle Type")
plt.ylabel("Electric Range (miles)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ---------------------------
# OBJECTIVE 5: Pie Chart
# ---------------------------
ev_type_counts = df_ev_clean["Electric Vehicle Type"].value_counts()
plt.pie(ev_type_counts.values, labels=ev_type_counts.index, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel"))
plt.title("Pie Chart: Distribution of Electric Vehicle Types")
plt.tight_layout()
plt.show()

# ---------------------------
# OBJECTIVE 6: Heatmap
# ---------------------------
corr_data = df_ev_clean[["Model Year", "Electric Range", "Base MSRP"]].dropna()
corr = corr_data.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap: Correlation Between Numeric Features")
plt.tight_layout()
plt.show()

# ---------------------------
# OBJECTIVE 7: Violin Plot
# ---------------------------
sns.violinplot(data=df_plot, x="Electric Vehicle Type", y="Base MSRP", palette="Accent")
plt.title("Violin Plot: MSRP by Electric Vehicle Type")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ---------------------------
# OBJECTIVE 8: Top 10 Utilities
# ---------------------------
top_utilities = df_ev_clean["Electric Utility"].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(y=top_utilities.index, x=top_utilities.values, palette="coolwarm")
plt.title("Top 10 Electric Utilities by EV Count")
plt.xlabel("Number of Vehicles")
plt.ylabel("Electric Utility")

for i, v in enumerate(top_utilities.values):
    plt.text(v + 500, i, str(v), va='center')

plt.tight_layout()
plt.show()

# ---------------------------
# OBJECTIVE 9: Histogram
# ---------------------------
sns.histplot(df_ev_clean["Electric Range"], bins=30, kde=True)
plt.title("Histogram: Distribution of Electric Range")
plt.xlabel("Electric Range (miles)")
plt.tight_layout()
plt.show()

# ---------------------------
# OBJECTIVE 10: Line Plot of Average MSRP by Year
# ---------------------------
avg_price_by_year = df_plot.groupby("Model Year")["Base MSRP"].mean()
avg_price_by_year.plot(marker='o', color='teal')
plt.title("Line Graph: Average MSRP by Model Year")
plt.ylabel("Average MSRP ($)")
plt.xlabel("Model Year")
plt.tight_layout()
plt.show()

# ===========================
# ENHANCEMENT: STATS & TESTS
# ===========================

# Descriptive Statistics
print("Descriptive Statistics:\n", df_ev_clean[["Model Year", "Electric Range", "Base MSRP"]].describe())

# Shapiro-Wilk Test for Normality
stat, p = shapiro(df_ev_clean["Electric Range"].sample(n=500, random_state=1))
print(f"\nShapiro-Wilk Test: Statistics={stat:.4f}, p-value={p:.4f}")
print("Normally distributed" if p > 0.05 else "Not normally distributed")

# t-test between BEV and PHEV MSRP
bev = df_ev_clean[df_ev_clean["Electric Vehicle Type"] == "Battery Electric Vehicle (BEV)"]["Base MSRP"]
phev = df_ev_clean[df_ev_clean["Electric Vehicle Type"] == "Plug-in Hybrid Electric Vehicle (PHEV)"]["Base MSRP"]
t_stat, p_val = ttest_ind(bev.dropna(), phev.dropna())
print(f"\nt-test: t-stat={t_stat:.4f}, p-value={p_val:.4f}")
print("Significant difference" if p_val < 0.05 else "No significant difference")

# Chi-Squared Test: EV Type vs Make
contingency_table = pd.crosstab(df_ev_clean["Electric Vehicle Type"], df_ev_clean["Make"])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-Squared Test: chi2={chi2:.2f}, p-value={p:.4f}")

# VIF Analysis
df_vif = df_ev_clean[["Model Year", "Electric Range", "Base MSRP"]].dropna()
X = add_constant(df_vif)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factor (VIF):\n", vif_data)

# ===========================
# PROBABILITY DISTRIBUTIONS
# ===========================

# Uniform Distribution
x_uniform = np.linspace(0, 1, 100)
plt.plot(x_uniform, [1]*100, label="Uniform PDF")
plt.title("Uniform Distribution")
plt.xlabel("x")
plt.ylabel("Probability")
plt.show()

# Normal Distribution
x_norm = np.linspace(-4, 4, 1000)
plt.plot(x_norm, norm.pdf(x_norm), label="Normal PDF")
plt.title("Normal Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.show()

# Binomial Distribution
n, p = 10, 0.5
x_binom = np.arange(0, 11)
plt.bar(x_binom, binom.pmf(x_binom, n, p))
plt.title("Binomial Distribution (n=10, p=0.5)")
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.show()

# Poisson Distribution
mu = 3
x_poisson = np.arange(0, 10)
plt.bar(x_poisson, poisson.pmf(x_poisson, mu))
plt.title("Poisson Distribution (mu=3)")
plt.xlabel("Events")
plt.ylabel("Probability")
plt.show()

# ===========================
# A/B TESTING SIMULATION
# ===========================
np.random.seed(42)
group_A = np.random.normal(loc=35000, scale=3000, size=100)
group_B = np.random.normal(loc=37000, scale=3000, size=100)
t_stat, p_val = ttest_ind(group_A, group_B)
print(f"\nA/B Testing (Simulated MSRP): t-stat={t_stat:.2f}, p-value={p_val:.4f}")
print("Significant difference between A and B" if p_val < 0.05 else "No significant difference between A and B")