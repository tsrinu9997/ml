# Feature creation

This folder adapts the feature-creation topics from the [referenced Section 17 GitHub examples](https://github.com/solegalli/feature-engineering-for-machine-learning/tree/main/Section-17-Feature-creation). Feature creation means combining existing columns or transforming them into new columns that express a relationship the model can use more easily.

## Why create new features?

Raw columns are often incomplete descriptions of the real problem. For a house, living area and basement area are useful separately, but total usable area may be more meaningful. For a customer, spend and number of purchases are useful separately, but average spend per order may be a stronger signal.

Feature creation can:

- add domain knowledge to a model;
- expose ratios, totals, gaps, and interactions;
- represent non-linear relationships;
- improve accuracy with a simpler estimator;
- make a result easier to explain to a business stakeholder.

Feature creation is not a guarantee of improvement. New columns can be redundant, noisy, unstable, or leaky. Always compare with a baseline using validation.

## Notebooks

- [mathematical_features.ipynb](mathematical_features.ipynb): sum, product, count, mean, and standard deviation features.
- [relative_features.ipynb](relative_features.ipynb): subtraction and division features such as usable area and area ratios.
- [polynomial_features.ipynb](polynomial_features.ipynb): polynomial and interaction terms for non-linear relationships.
- [spline_features.ipynb](spline_features.ipynb): smooth piecewise curves that model non-linearity without a high-degree polynomial.
- [feature_creation_feature_engine.ipynb](feature_creation_feature_engine.ipynb): MathFeatures and RelativeFeatures inside a complete pipeline.
- [02-feature-engine/](02-feature-engine/): the dedicated Feature-engine subsection with separate MathFeatures, RelativeFeatures, and pipeline notebooks.

All examples use the repository’s Ames house-price data when available and include a path search for running from the project root or this folder.

## Main techniques

### Mathematical combinations

Create a new value from one or more columns:

- sum: total area or total spend;
- product: quantity multiplied by price;
- count: number of non-missing components;
- mean: average rating or average area;
- standard deviation: disagreement or variability between measurements.

Use combinations when the calculation has a clear business meaning. Avoid blindly creating every possible combination because the feature count can grow quickly.

### Relative features

Subtraction measures a gap or balance:

    total_area - basement_area = above_ground_area

Division measures a rate or proportion:

    living_area / lot_area = land_utilisation_ratio

Protect division features from zero denominators and document the unit. A ratio can be more comparable across different-sized entities, but it can also become unstable when the denominator is very small.

### Polynomial and interaction features

Polynomial features add powers and interactions:

    x, x², x³, x1*x2

They let linear regression represent curves and combinations. Degree 2 is often a reasonable starting point. Higher degrees can create many columns and overfit.

### Splines

Splines join several low-degree polynomial pieces at learned or chosen knots. They create a smooth curve with local flexibility and are often more stable and interpretable than a single high-degree polynomial.

## Advantages

- Encodes subject-matter knowledge directly.
- Can reveal relationships hidden across columns.
- Ratios and differences can reduce dependence on units or scale.
- Polynomial terms give linear models non-linear capacity.
- Splines provide smooth non-linearity with controlled flexibility.
- Features can be inspected and explained individually.

## Limitations

- More columns can increase computation and overfitting.
- Ratios can be unstable near zero.
- Polynomial expansion can create multicollinearity.
- Splines require a sensible number and placement of knots.
- A feature calculated from future information causes leakage.
- A formula that works for one dataset may not generalize to another.

## Safe workflow

1. Start with a baseline model and record its validation score.
2. Write down the business question represented by the new feature.
3. Check units, missing values, zero denominators, and impossible results.
4. Split the data before learning any target-dependent or data-dependent parameters.
5. Create deterministic features with a reusable function or pipeline transformer.
6. Fit polynomial and spline parameters on training data only.
7. Compare metrics, residuals, and feature stability with the baseline.
8. Keep only features that improve generalisation or interpretation.

## Applications

- **House prices:** total finished area, age at sale, living-area ratio, and total bathrooms.
- **Retail:** average order value, discount rate, basket size, and product margin.
- **Finance:** debt-to-income ratio, utilisation rate, and payment gap.
- **Healthcare:** change from baseline, dose per body weight, and variability across readings.
- **Manufacturing:** yield ratio, temperature difference, and sensor interaction.
- **Marketing:** spend per campaign, recency, frequency, and monetary value.

## Interview answer

> I create a feature when it represents a meaningful relationship that the raw columns do not express directly. I validate the formula, protect edge cases such as zero denominators, fit learned transformations inside the training pipeline, and compare against a baseline. More features are not automatically better; I watch for leakage, multicollinearity, and overfitting.

## Run the notebooks

    python -m pip install pandas numpy matplotlib seaborn scikit-learn feature-engine jupyter
    jupyter notebook
