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

Use `Winsorizer` when extreme values should remain in the data but have limited influence. Use `ArbitraryOutlierCapper` when a business, physical, or measurement boundary is known. Use `OutlierTrimmer` when extreme observations should be excluded from the analysis and you have a defensible reason to remove rows.

The three transformers are part of Feature-engine's outlier capping and removal functionality. Install dependencies in the notebook kernel if needed:

```bash
pip install feature-engine pandas numpy matplotlib seaborn scikit-learn
```
