# Machine-learning pipelines: one repeatable preprocessing recipe

## What is a pipeline?

A pipeline is an ordered chain of data-preparation and modelling steps. Think of an assembly line: every row enters at the beginning, passes through the same stations, and leaves as a prediction.

    raw data -> imputation -> feature creation -> encoding -> scaling -> model

The pipeline stores the rules learned during fitting, so a future row receives the same treatment as a training row.

## Why do we need pipelines?

Without a pipeline, it is easy to apply a transformation to training data but forget it at prediction time, or to calculate a median and category average using validation data. Pipelines reduce these errors and make the complete experiment reproducible.

## Advantages

- Prevents preprocessing leakage during cross-validation.
- Guarantees that training and prediction use the same order of operations.
- Makes experiments easier to compare.
- Allows preprocessing and the estimator to be saved as one object.
- Works naturally with grid search and cross-validation.
- Gives a clear hand-off from notebook exploration to deployment.

## When should we use one?

Use a pipeline whenever a model has learned preprocessing, including numerical imputation, category grouping or encoding, scaling, outlier limits, discretisation, target encoding, and domain feature creation that must be repeated at prediction time.

A deterministic feature such as age from two dates may be computed before the pipeline, but putting it in a reusable transformer is safer for production.

## A simple example

    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ('imputer', MeanMedianImputer()),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ('scaler', StandardScaler()),
        ('regressor', Lasso())
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

The test set is passed to predict only after fitting is complete. Every learned step uses the training data.

## Recommended order

1. Remove identifiers and post-outcome fields.
2. Split into training and test data.
3. Create deterministic domain features.
4. Add missing indicators where absence matters.
5. Impute numerical and categorical missing values.
6. Group rare labels.
7. Encode categories.
8. Cap or trim outliers if justified.
9. Discretise or transform skewed features if useful.
10. Scale features for algorithms that depend on magnitude.
11. Fit the estimator.
12. Evaluate on untouched validation or test data.

Some steps can be reordered. Document the reason rather than following a fixed recipe blindly.

## Cross-validation and leakage

When cross-validation is used, the entire pipeline must be passed to cross_validate or GridSearchCV. Each fold should fit its own imputer, encoder, scaler, discretiser, and model.

Target encoding needs extra care because it uses y. A leakage-safe target encoder uses training folds to calculate statistics for validation rows. Never calculate a category target mean from the same row's answer and then train on that row.

## Notebooks in this folder

- [adult-income-with-feature-engine.ipynb](adult-income-with-feature-engine.ipynb): classification for income prediction.
- [create-new-features-with-feature-engine.ipynb](create-new-features-with-feature-engine.ipynb): domain-based wine feature creation.
- [predict-house-price-with-feature-engine.ipynb](predict-house-price-with-feature-engine.ipynb): imputation, encoding, discretisation, scaling, and Lasso regression.

## Applications

- **Batch scoring:** transform and predict a large file with the same saved recipe.
- **Real-time APIs:** receive one new record and call pipeline.predict.
- **Model comparison:** swap only the final estimator while keeping preprocessing consistent.
- **Hyperparameter tuning:** search preprocessing and model parameters together.
- **Monitoring:** compare production feature distributions with training distributions.
- **Reproducible research:** rerun the same steps on a new split or dataset.

## Interview answer

> I put all learned preprocessing and the estimator in one scikit-learn pipeline. During cross-validation, each fold fits the complete pipeline from scratch, which prevents leakage. The fitted pipeline can then be saved and used consistently for future predictions.

## Common mistakes

- Fitting an encoder, scaler, or imputer before splitting the data.
- Applying a notebook-only transformation that is not available in production.
- Changing column order or category policy between training and inference.
- Including IDs or post-outcome fields.
- Reporting only training performance.
