# Feature creation with Feature-engine

This subsection recreates the Feature-engine portion of Section 17. Feature-engine provides pandas-friendly transformers that can be fitted, inspected, and placed inside scikit-learn pipelines.

## Files

- [math_features.ipynb](math_features.ipynb): MathFeatures for sums, means, products, standard deviations, and counts.
- [relative_features.ipynb](relative_features.ipynb): RelativeFeatures for subtraction and division against reference columns.
- [feature_creation_pipeline.ipynb](feature_creation_pipeline.ipynb): both transformers inside a regression pipeline.

## Why use Feature-engine?

Manual formulas are easy to write but easy to forget at prediction time. A transformer stores the selected variables and operation, adds features consistently, and can be fitted inside each cross-validation fold.

The current API uses:

- MathFeatures for mathematical combinations of variables.
- RelativeFeatures for subtraction and division relative to reference variables.

## Safe practice

1. Define a feature formula with a clear business meaning.
2. Fit data-dependent transformations on training data only.
3. Protect ratios from zero denominators.
4. Keep the original columns when their absolute values still matter.
5. Place creation steps and the estimator inside one Pipeline.
6. Compare against a raw-feature baseline.

## Interview answer

> Feature-engine turns feature formulas into reusable sklearn-compatible transformers. I use MathFeatures for combinations such as sums or means and RelativeFeatures for gaps or ratios. I fit them inside the training pipeline, protect division by zero, and validate whether the new features improve generalisation.

## Official documentation

- [Feature-engine creation user guide](https://feature-engine.trainindata.com/en/latest/user_guide/creation/index.html)
- [MathFeatures API](https://feature-engine.trainindata.com/en/latest/api_doc/creation/MathFeatures.html)
- [RelativeFeatures API](https://feature-engine.trainindata.com/en/latest/api_doc/creation/RelativeFeatures.html)
