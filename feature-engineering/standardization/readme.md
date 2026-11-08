# Standardization: expressing values as z-scores

## What is standardization?

Standardization converts a numerical feature into its distance from the training average, measured in standard deviations.

    z = (value - training_mean) / training_standard_deviation

After fitting, the training feature has an average close to 0 and a standard deviation close to 1. A z-score of 2 means the observation is approximately two standard deviations above the training average.

## Why do we need it?

Many algorithms are affected by feature magnitude. A variable measured in dollars can have values much larger than a variable measured in years. Standardization makes numerical scales comparable so that a large unit does not automatically receive more influence.

It is particularly useful when the model uses distances, dot products, regularisation penalties, or gradient descent.

## When is standardization useful?

Common examples include linear and logistic regression with regularisation, support-vector machines, k-nearest neighbours, k-means clustering, principal component analysis, neural networks, and gradient-based estimators.

Decision trees and most tree ensembles usually do not need standardization because their splits depend on order and thresholds, not distance from zero.

## Advantages

- Features are on a comparable scale.
- The output has a useful statistical interpretation.
- Regularisation treats coefficients more fairly.
- PCA and distance calculations are less dominated by unit choice.
- It does not require every feature to have a fixed 0–1 range.

## Limitations

- The mean and standard deviation are affected by extreme values.
- A highly skewed feature may remain skewed after standardization.
- A future value can have a very large positive or negative z-score.
- The result is less directly interpretable in original units.
- Binary indicator columns usually do not need to be standardized like continuous measurements.

For severe skew, investigate a log or power transformation, a robust scaler, or a domain-specific approach before standardizing.

## Standardization versus normalization

Normalization usually maps values to 0–1 using the minimum and maximum. Standardization centres values around zero and measures spread using the standard deviation. Choose based on the model and data, then compare alternatives using validation.

## Step-by-step workflow

1. Select continuous numerical features.
2. Inspect outliers and skewness.
3. Split into train and test.
4. Fit StandardScaler on training features only.
5. Transform all later data with the fitted scaler.
6. Keep categorical encoding and numerical scaling in separate, clearly named pipeline steps.
7. Check validation metrics and coefficient stability.

Example:

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

## Notebooks in this folder

- [standaardization.ipynb](standaardization.ipynb): the original notebook name is retained for compatibility.
- [standardization1.ipynb](standardization1.ipynb): an additional standardization example.
- [Social_Network_Ads.csv](Social_Network_Ads.csv): the dataset used by the examples.

## Applications

- **Credit scoring:** put income, age, and account measures on comparable scales.
- **PCA:** prevent high-unit measurements from dominating principal components.
- **Marketing segmentation:** improve distance calculations between customer profiles.
- **Medical modelling:** compare lab measurements with different units.
- **Regularized regression:** make the penalty comparable across coefficients.

## Interview answer

> Standardization subtracts the training mean and divides by the training standard deviation. It is useful for models that depend on scale, distance, or regularisation. I fit the scaler within the training fold and remember that standardization does not remove skewness or make outliers disappear.

## Common mistakes

- Computing mean and standard deviation from the full dataset.
- Scaling categorical labels as if they were continuous measurements.
- Assuming tree models require scaling.
- Forgetting to apply exactly the same fitted scaler at inference time.
