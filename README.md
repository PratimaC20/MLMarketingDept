# Customer Segmentation & Behavior Analysis

---

## 1. Project Overview

Understanding customer spending habits, payment behaviors, and credit utilization is essential for targeted marketing, credit risk management, and financial product design. 

This project processes transactional and behavioral credit card data across **8,950 customer records** to perform end-to-end data processing, exploratory analysis, and unsupervised machine learning. The primary goal is to segment the customer base into distinct, actionable financial personas using **K-Means Clustering** and **Principal Component Analysis (PCA)** for dimensionality reduction.

---

## 2. Dataset Description

The dataset contains **17 quantitative financial tracking features** tracking customer behavior over their tenure:

| Feature Name | Description |
| :--- | :--- |
| `BALANCE` | Total balance amount remaining in customer's account |
| `BALANCE_FREQUENCY` | Score between 0 and 1 indicating how frequently balance is updated |
| `PURCHASES` | Total purchase amount made from the account |
| `ONE OFF_PURCHASES` | Maximum purchase amount completed in a single transaction |
| `INSTALLMENTS_PURCHASES` | Total amount of purchases done in installments |
| `CASH_ADVANCE` | Total cash advance taken by the user |
| `PURCHASES_FREQUENCY` | Score between 0 and 1 indicating overall purchase frequency |
| `ONE OFF_PURCHASES_FREQUENCY` | Score between 0 and 1 indicating one-off purchase frequency |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Score between 0 and 1 indicating installment purchase frequency |
| `CASH_ADVANCE_FREQUENCY` | Score between 0 and 1 indicating frequency of cash advances |
| `CASH_ADVANCE_TRX` | Total number of cash advance transactions |
| `PURCHASES_TRX` | Total number of purchase transactions |
| `CREDIT_LIMIT` | Maximum credit card limit assigned to the user |
| `PAYMENTS` | Total amount paid by the user toward their bill |
| `MINIMUM_PAYMENTS` | Minimum payment amount executed by the user |
| `PRC_FULL_PAYMENT` | Percentage of full statement balance paid by the user |
| `TENURE` | Tenure of credit card service for the user (in months) |

---

## 3. Data Preprocessing & Cleaning

1. **Identifier Removal:** Dropped `CUST_ID` from the dataset as it acts solely as an index with zero predictive power.
2. **Missing Value Imputation:**
   * `MINIMUM_PAYMENTS`: Imputed 313 missing records using the feature mean (**$864.21**).
   * `CREDIT_LIMIT`: Imputed 1 missing record using the feature mean (**$4,494.45**).
3. **Data Integrity:** Verified zero duplicate records and achieved 100% data completeness across all 17 features.
4. **Feature Scaling:** Prepared feature vectors using `StandardScaler` to handle variance differences across numerical features prior to clustering.

---

## 4. Comprehensive Analysis & Findings

| Metric / Feature | Mean Value | Max Value |
| :--- | :--- | :--- |
| Balance | $1,564.47 | $19,043.14 |
| Purchases | $1,003.20 | $49,039.57 |
| One-Off Purchases | $592.44 | $40,761.25 |
| Installments Purchases | $411.07 | $22,500.00 |
| Cash Advance | $978.87 | $47,137.21 |
| Credit Limit | $4,494.45 | $30,000.00 |
| Payments | $1,733.14 | $50,721.48 |
| Minimum Payments | $864.21 | $76,406.21 |
| Percent Full Payment | 15.37% | 100.00% |
### Key Insights
* **Revolving Balances:** On average, customers only pay off their full balance **15.37%** of the time. This indicates high balance revolving behavior, generating substantial interest revenue for the institution.
* **Installment Behavior vs. One-Off:** While average One-Off spend ($592.44) is higher than Installment spend ($411.07), Purchase Installment Frequency (0.36) is nearly double One-Off Frequency (0.20), showing heavy reliance on recurring payment structures.
* **Skewed Distribution:** Significant variance between 75th percentile values and maximum values across `PURCHASES`, `CASH_ADVANCE`, and `PAYMENTS` highlights a small segment of ultra-high spenders and cash-dependent accounts.

---

## 5. Machine Learning Pipeline (Clustering)

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Load and Clean Dataset
df = pd.read_csv('Marketing_data.csv').drop(columns=['CUST_ID'])
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].mean(), inplace=True)
df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean(), inplace=True)

# 2. Feature Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 3. Dimensionality Reduction (PCA)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# 4. K-Means Clustering Assignment
kmeans = KMeans(n_clusters=4, random_state=42)
df['CLUSTER'] = kmeans.fit_predict(scaled_data)
```
## 6. Customer Personas & Marketing Strategy

* **Cluster 0 — Transactors (Low Balance, High Spend, High Full Payments)**
  * *Strategy:* Promote premium rewards, higher cashback tiers, and card upgrades to maximize transaction volumes.
* **Cluster 1 — Revolvers (High Balance, Moderate Spend, Low Full Payments)**
  * *Strategy:* Offer flexible EMI conversion options and low-interest balance transfer programs to manage credit default risk while maintaining interest yields.
* **Cluster 2 — Cash Seekers (High Cash Advance, Low Purchases, Lower Credit Limits)**
  * *Strategy:* Provide structured liquidity products or short-term personal loan alternatives while lowering cash advance credit boundaries to mitigate risk.
* **Cluster 3 — VIPs / High-Rollers (Top Credit Limits, High Spend, High Payments)**
  * *Strategy:* Deliver premium concierge services, exclusive merchant partnerships, and proactive credit limit expansion to retain high-value accounts.

---

## 7. Tech Stack

* **Programming Language:** Python
* **Data Processing & Manipulation:** `pandas`, `numpy`
* **Machine Learning & Preprocessing:** `scikit-learn` (`StandardScaler`, `PCA`, `KMeans`)
* **Data Visualization:** `matplotlib`, `seaborn`
* **Development Environment:** Jupyter Notebook
