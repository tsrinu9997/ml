# Normalization: putting numerical features on a common range

## What is normalization?

Normalization rescales a numerical feature to a fixed interval, most often 0 to 1. It does not change the order of observations; it changes their units.

For min–max normalization:

    normalized_value = (value - training_min) / (training_max - training_min)

The smallest training value becomes 0 and the largest becomes 1. A value halfway between them becomes approximately 0.5.

## Why is it needed?

Suppose one feature is age in years and another is annual salary in dollars. A distance-based algorithm may treat salary as more important only because its numbers are larger. Normalization gives features a comparable numerical range.

It is especially useful when the algorithm calculates distances, dot products, or gradient updates.

## When should we use it?

Normalization is a common starting point for k-nearest neighbours, k-means clustering, neural networks, gradient-based optimisation, and algorithms where bounded inputs make training easier. Tree-based models usually do not require scaling because they compare features with thresholds.

## Advantages

- Creates an intuitive, bounded range.
- Prevents large measurement units from dominating distances.
- Can improve optimisation stability for neural networks.
- Makes plots and feature comparisons easier.
- Preserves the rank order of values.

## Limitations

- The minimum and maximum can be strongly affected by outliers.
- A future value outside the training range may become less than 0 or greater than 1.
- The output is less interpretable in original business units.
- It does not make a non-linear relationship linear.
- It is not a replacement for outlier investigation.

If outliers are genuine, consider a robust scaler, a suitable transformation, or carefully justified capping before normalization.

## Normalization versus standardization

| Question | Normalization | Standardization |
|---|---|---|
| Typical output | 0 to 1 | Mean 0 and standard deviation 1 |
| Parameters learned | Minimum and maximum | Mean and standard deviation |
| Best known use | Bounded, distance-based inputs | Linear models, SVM, PCA, gradient methods |
| Main caution | Extreme values squeeze the range | Extreme values affect mean and spread |

Neither approach is automatically better. Test the method with the chosen model and validation process.

## Step-by-step workflow

1. Identify numerical features that have incompatible units.
2. Inspect their distributions and check for outliers.
3. Split the data into training and test sets.
4. Fit the normalizer on training data only.
5. Transform training, validation, test, and future records with the same fitted object.
6. Confirm the new ranges and check for unexpected values outside the interval.
7. Compare model performance with and without normalization.

Example:

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

The test set must use transform, never fit_transform.

## Notebooks in this folder

- [normalization.ipynb](normalization.ipynb): normalization with Social Network Ads data.
- [normalization_winedata.ipynb](normalization_winedata.ipynb): normalization analysis with wine data.
- [Social_Network_Ads.csv](Social_Network_Ads.csv): age, estimated salary, and purchase outcome.
- [wine_data.csv](wine_data.csv): numerical wine measurements.

## Applications

- **Customer similarity:** compare age and salary without salary overwhelming the distance.
- **Recommendation systems:** put product measurements on comparable ranges.
- **Image and sensor data:** keep inputs in a stable interval for neural networks.
- **Clustering:** prevent one high-unit column from defining all clusters.
- **Optimisation:** help gradient-based models move through the loss surface smoothly.

## Interview answer

> Normalization commonly uses min–max scaling to map a numerical feature to a fixed interval. I fit the minimum and maximum on the training data only. It is useful for distance- and gradient-based models, but I check for outliers because extreme values can compress most observations into a small part of the range.

## Common mistakes

- Fitting the scaler before the train/test split.
- Scaling the target without documenting how predictions return to original units.
- Assuming every algorithm needs scaling.
- Treating normalization as a way to remove outliers.
