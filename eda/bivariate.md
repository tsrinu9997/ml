# Bivariate Analysis in Banking and Financial Markets

**Bivariate analysis**—the simultaneous examination of two variables to determine their empirical relationship, correlation, dependency, and predictive interaction—is a cornerstone of quantitative finance, commercial banking, and capital markets. 

While univariate analysis establishes baseline feature health and individual distributions, bivariate analysis models how variables move together: independent risk driver vs. dependent outcome, macro factor vs. asset price, or borrower attribute vs. probability of default.

Here is a breakdown of how bivariate analysis is applied across key banking and financial market domains:

---

## 1. Credit Risk Modeling and Loan Underwriting

In retail and commercial lending, risk analysts must determine how applicant and loan-level attributes relate to repayment behavior before assembling multi-variable credit scorecards or machine learning default models.

* **Multicollinearity and Feature Correlation Screening:** Before training logistic regression scorecards or survival analysis models, risk teams calculate pairwise correlation matrices (Pearson for linear, Spearman for non-linear rank correlations) across candidate predictors (e.g., Debt-to-Income vs. Credit Score, Loan Amount vs. Annual Income). Highly correlated feature pairs ($r > 0.7$) are identified to eliminate redundancy and prevent variance inflation in model coefficients.
* **Continuous vs. Binary Default Analysis (Log-Odds Linearity):** Bivariate plots (such as mean default rate across binned credit score deciles or age brackets) test whether risk increases monotonically. This ensures that scorecards satisfy linear log-odds assumptions or reveals whether non-linear spline transformations are necessary.
* **Contingency Tables & Chi-Square Tests ($\chi^2$):** For categorical variables (e.g., Employment Type vs. Default Status, or Loan Purpose vs. Charge-off), bivariate contingency tables (`pd.crosstab`) and Chi-Square tests of independence determine whether categorical segments exhibit statistically significant differences in default rates.

---

## 2. Capital Markets, Asset Pricing, and Quantitative Trading

Investment banks, hedge funds, and asset managers rely on bivariate statistical relationships to value securities, construct hedges, and identify mispriced assets.

* **Capital Asset Pricing Model (CAPM) & Market Beta ($\beta$):** Bivariate linear regression between an individual security’s excess return ($R_i$) and the benchmark market return ($R_m$) estimates the stock's **Beta** ($\beta = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}$) and alpha ($\alpha$). Analysts use scatterplots with fitted regression lines to visualize systematic risk exposure and tracking error.
* **Pairs Trading and Cointegration Analysis:** Quantitative desks analyze the bivariate price ratio and spread between two historically correlated equities or commodities. When the bivariate spread diverges beyond historical standard deviation thresholds, mean-reversion trading strategies are initiated.
* **Yield Curve Spread Analysis (2s10s Curve):** Fixed income traders track the bivariate relationship between short-term (2-year) and long-term (10-year) government bond yields. The slope (spread) is monitored continuously as an indicator of monetary policy expectations and economic recession forecasts.

---

## 3. Market Risk, Portfolio Optimization, and Hedging

Enterprise risk management functions within banks must measure cross-asset portfolio vulnerability and comply with Basel III / FRTB (Fundamental Review of the Trading Book) market risk mandates.

* **Covariance and Modern Portfolio Theory (MPT):** The foundation of Markowitz portfolio optimization is the bivariate covariance between every pair of portfolio assets. Bivariate relationships determine the diversification benefit: combining assets with negative or low positive correlations reduces total portfolio variance without sacrificing expected return.
* **Value at Risk (VaR) & Copula Modeling:** Parametric VaR models rely on pairwise correlation matrices. During market stress, linear correlation often breaks down; risk modelers use bivariate copulas to capture asymmetric tail dependence (e.g., assets becoming highly correlated during market crashes).
* **Cross-Hedging Ratios:** Commodity and FX trading desks use bivariate regression slopes to compute minimum-variance hedge ratios when hedging an illiquid financial exposure using liquid futures contracts.

---

## 4. Asset-Liability Management (ALM) and Liquidity Risk

Treasury and ALM teams in commercial banks oversee the interest rate risk in the banking book (IRRBB) and manage the spread between deposit liabilities and loan assets.

* **Deposit Beta Estimation:** Bivariate regression between central bank policy rate shifts (e.g., Federal Funds Rate changes) and bank deposit interest rate changes determines the bank’s **Deposit Beta**. Understanding this relationship is critical for forecasting Net Interest Margin (NIM) and liquidity runoff under rate-hiking cycles.
* **Prepayment Risk vs. Interest Rate Differentials:** Mortgage lenders analyze the bivariate relationship between prevailing market mortgage rates and prepayment speeds (Conditional Prepayment Rate - CPR). An inverse relationship models mortgage refinancing waves.

---

## 5. Fraud Detection and AML Surveillance

Transaction monitoring systems evaluate pairs of features to catch fraudulent behavior that looks benign when analyzed on a single-variable basis.

* **Amount vs. Velocity (Velocity Profiling):** A transaction of $5,000 might be normal on a univariate level, but when paired with transaction velocity (e.g., 5 transactions within 10 minutes), the bivariate profile triggers an immediate account freeze.
* **Customer Risk Rating vs. Transaction Spikes:** AML compliance cross-tabulates customer risk tiers (High, Medium, Low) against international wire transfer volumes to identify anomalies requiring Suspicious Activity Reports (SAR).

---

### Summary of Key Bivariate Techniques in Banking & Financial Markets

| Technique / Metric | Primary Financial Domain | Purpose / Banking Application |
| :--- | :--- | :--- |
| **Pearson & Spearman Correlation** | Credit Risk & Portfolio Mgmt | Screening feature pairs for multicollinearity; quantifying linear and monotonic co-movements. |
| **Bivariate OLS Regression (Slope & $R^2$)** | Capital Markets & ALM | Calculating CAPM Beta ($\beta$), factor loadings, and interest rate deposit betas. |
| **Contingency Tables & Chi-Square ($\chi^2$)** | Underwriting & AML Surveillance | Testing independence between categorical risk tiers (loan grade, occupation) and default/fraud. |
| **Grouped Boxplots & Barplots** | Risk Profiling & Scorecards | Visualizing credit loss severity (LGD) or recovery rates conditioned on collateral type or borrower class. |
| **Rolling Bivariate Correlation** | Trading Desks & Macro Research | Detecting structural regime shifts, contagion between markets, and breakdown of traditional hedging pairs. |
| **Yield Curve Differentials (Spreads)** | Fixed Income & Macro Modeling | Tracking economic cycles, term premiums, and yield curve inversion indicators. |
