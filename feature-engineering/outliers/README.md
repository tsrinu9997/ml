# Outlier handling with Feature-engine

This folder mirrors the three outlier examples listed in the Feature-engine examples repository:

- [Winsorizer.ipynb](Winsorizer.ipynb) detects statistical extremes and caps them at learned limits while keeping every row.
- [ArbitraryOutlierCapper.ipynb](ArbitraryOutlierCapper.ipynb) caps values at limits supplied by the analyst or a domain specification.
- [OutlierTrimmer.ipynb](OutlierTrimmer.ipynb) removes rows that are outside the selected statistical limits.

Titanic-specific versions are also included:

- [Winsorizer_titanic.ipynb](Winsorizer_titanic.ipynb)
- [ArbitraryOutlierCapper_titanic.ipynb](ArbitraryOutlierCapper_titanic.ipynb)
- [OutlierTrimmer_titanic.ipynb](OutlierTrimmer_titanic.ipynb)

The Titanic notebooks use `titanic.csv`, impute missing numeric age and fare values using training medians, keep `survived` separate as the target, and exclude post-event fields such as `boat` and `body`.

The notebooks use `feature-engineering/imputation/houseprice.csv`. Each notebook searches for that file from either the project root or the outliers folder. They fit outlier rules on a training split, apply the fitted rule to the test split, and include plots and checks showing what changed.

## Choosing a method

## What is an outlier?

An outlier is an observation that is unusually far from the normal pattern. It may be a typing error, a sensor failure, a valid rare event, or a member of a different population. Statistical unusualness is a signal to investigate, not automatic proof that the row should be deleted.

For example, a very large house may be a genuine luxury home. A fare that is ten times larger because an extra zero was typed is a data-quality error. The correct treatment is different in each case.

## Why handle outliers?

Extreme values can:

- pull the mean and standard deviation away from typical values;
- dominate a linear regression loss;
- distort normalization and standardization;
- create misleading plots and correlations;
- cause a distance-based model to treat a few rows as very important.

Tree models are often more tolerant, but outliers can still affect splits, metrics, and business decisions.

## How to detect them

- **Boxplot/IQR:** flag values below Q1 - 1.5 x IQR or above Q3 + 1.5 x IQR.
- **Z-score:** flag values several standard deviations from the mean.
- **Quantiles:** inspect the highest and lowest percentiles.
- **Domain limits:** use physically possible or business-approved boundaries.
- **Visual checks:** combine histograms, scatterplots, and row-level investigation.

## Choosing a treatment

| Treatment | Meaning | Advantages | Limitation |
|---|---|---|---|
| Keep | The extreme value is valid and useful | Preserves information | May influence a sensitive model |
| Winsorize/cap | Replace extremes with learned limits | Keeps every row and reduces influence | Changes the original value |
| Arbitrary cap | Apply expert-defined limits | Easy to explain and audit | Requires trustworthy domain limits |
| Trim | Remove the row | Useful for impossible or out-of-scope records | Loses data and may bias the sample |
| Transform | Use log or another scale | Reduces skew while keeping rows | Interpretation changes |
| Robust model | Choose a less sensitive estimator | Avoids arbitrary deletion | Does not repair bad data |

## Step-by-step workflow

1. Plot the variable and locate extreme observations.
2. Check the source record, units, and business meaning.
3. Decide whether the value is an error, a valid rare case, or another population.
4. Split data before learning statistical limits.
5. Fit the capper or trimming rule on training data only.
6. Apply the fitted rule to validation, test, and future data.
7. Compare distributions and model metrics before and after treatment.
8. Record how many values were capped or rows removed.

## Advantages of Feature-engine outlier transformers

- The rule is stored and reused consistently.
- They work naturally in a scikit-learn pipeline.
- Training limits can be inspected and documented.
- Capping and trimming can be compared without rewriting the workflow.

## Applications

- **Finance:** unusually large transactions or account balances.
- **Healthcare:** impossible measurements or rare but real clinical values.
- **Manufacturing:** sensor spikes and equipment faults.
- **Retail:** unusually large orders or returns.
- **House prices:** extreme areas, prices, or lot sizes.
- **Web analytics:** bot traffic and abnormal session lengths.

## Interview answer

> I investigate whether an outlier is an error or a valid rare observation before changing it. If valid but influential, I may cap it or use a robust model. If impossible, I correct or remove it. Any statistical limits are learned on training data only, and I report how many records were affected.

## Common mistakes

- Deleting every point beyond 1.5 x IQR without checking context.
- Calculating limits on the full dataset.
- Treating a rare but important customer as a mistake.
- Forgetting that capping changes the data distribution.
- Removing outliers from the test set just to improve a score.

Use `Winsorizer` when extreme values should remain in the data but have limited influence. Use `ArbitraryOutlierCapper` when a business, physical, or measurement boundary is known. Use `OutlierTrimmer` when extreme observations should be excluded from the analysis and you have a defensible reason to remove rows.

The three transformers are part of Feature-engine's outlier capping and removal functionality. Install dependencies in the notebook kernel if needed:

```bash
pip install feature-engine pandas numpy matplotlib seaborn scikit-learn
```
