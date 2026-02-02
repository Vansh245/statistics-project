
# ======================================================
# FOOD DELIVERY DATA ANALYSIS PROJECT
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# -----------------------
# Set seed and sample size
# -----------------------
np.random.seed(42)
n = 500

# -----------------------
# Create dataset
# -----------------------
data = pd.DataFrame({
    "order_id": range(1, n + 1),
    "zone": np.random.choice(["North", "South", "East", "West"], n),
    "restaurant_type": np.random.choice(
        ["Fast Food", "Cafe", "Fine Dining", "Cloud Kitchen"], n),
    "payment_mode": np.random.choice(
        ["UPI", "Card", "Cash"], n, p=[0.5, 0.3, 0.2]),
    "delivery_time": np.random.normal(35, 10, n).clip(10),
    "bill_amount": np.random.normal(600, 250, n).clip(150),
    "discount_percent": np.random.randint(0, 40, n),
    "customer_rating": np.random.randint(1, 6, n),
    "cancelled": np.random.choice(["Yes", "No"], n, p=[0.12, 0.88])
})

# -----------------------
# Data preprocessing
# -----------------------
data["cancelled_binary"] = data["cancelled"].map({"Yes": 1, "No": 0})

# -----------------------
# Exploratory Data Analysis
# -----------------------
print("Frequency Distribution (Restaurant Type):")
print(data["restaurant_type"].value_counts())

data["zone"].value_counts().plot(kind="bar")
plt.title("Orders by Delivery Zone")
plt.show()

plt.hist(data["delivery_time"], bins=10)
plt.title("Delivery Time Distribution")
plt.show()

data["payment_mode"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Payment Mode Share")
plt.ylabel("")
plt.show()

data.groupby("discount_percent")["delivery_time"].mean().plot(marker="o")
plt.title("Avg Delivery Time vs Discount")
plt.show()

sns.boxplot(x="restaurant_type", y="bill_amount", data=data)
plt.title("Bill Amount by Restaurant Type")
plt.show()

# -----------------------
# Statistics
# -----------------------
mean_dt = data["delivery_time"].mean()
median_dt = data["delivery_time"].median()
mode_dt = data["delivery_time"].mode()[0]

weighted_mean_bill = np.average(
    data["bill_amount"],
    weights=data["discount_percent"]
)

combined_mean_ns = data.loc[
    data["zone"].isin(["North", "South"]),
    "delivery_time"
].mean()

# -----------------------
# Correlation
# -----------------------
pearson_corr, _ = stats.pearsonr(
    data["discount_percent"],
    data["bill_amount"]
)

spearman_corr, _ = stats.spearmanr(
    data["delivery_time"],
    data["customer_rating"]
)

# -----------------------
# Regression
# -----------------------
slope, intercept, r, p, se = stats.linregress(
    data["discount_percent"],
    data["bill_amount"]
)

predicted_bill_25 = intercept + slope * 25

# -----------------------
# Probability
# -----------------------
P_cancelled = data["cancelled_binary"].mean()

P_upi_not_cancelled = len(
    data[(data["payment_mode"] == "UPI") &
         (data["cancelled"] == "No")]
) / len(data)

binomial_prob = stats.binom.pmf(2, 6, P_cancelled)
poisson_prob = stats.poisson.pmf(5, mu=3)

mu = data["delivery_time"].mean()
sigma = data["delivery_time"].std()
prob_gt_50 = 1 - stats.norm.cdf(50, mu, sigma)

# -----------------------
# Sampling
# -----------------------
simple_sample = data.sample(100, random_state=1)

stratified_sample = (
    data.groupby("zone", group_keys=False)
        .apply(lambda x: x.sample(25), include_groups=False)
)

# -----------------------
# Central Limit Theorem
# -----------------------
sample_means = [
    data["bill_amount"].sample(40).mean()
    for _ in range(1000)
]

plt.hist(sample_means, bins=30)
plt.title("CLT Demonstration (Bill Amount)")
plt.show()

# -----------------------
# Standard Error
# -----------------------
standard_error = sigma / np.sqrt(len(data))

print("Project executed successfully.")
