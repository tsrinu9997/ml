# Imputation: handling missing values

## What is imputation?

Imputation means filling an empty value with a reasonable replacement. A blank cell is not a number, so many machine-learning algorithms cannot use it directly. Imputation turns an incomplete table into a usable table while trying to preserve the information in the original data.

Think of a school form with one missing mark. We can fill it with the class median, record that the mark was missing, or ask for the value again. The right choice depends on why the value is missing and what the column means.

## Why do we need it?

Real datasets contain missing values because a customer skipped a question, a sensor stopped working, a field did not apply, a system failed, or a value was lost during data integration. Most models cannot calculate with an empty value. Deleting incomplete rows may remove useful observations or create a biased sample.

## Common imputation methods

| Method | How it works | Good starting situation | Main limitation |
|---|---|---|---|
| Mean | Fill with the average | Roughly symmetric numerical data | Sensitive to outliers |
| Median | Fill with the middle value | Skewed data or data with outliers | Hides natural variation |
| Mode | Fill with the most common category | Categorical columns | Can make one category too common |
| Constant | Use a value such as Missing or 0 | Missingness has a clear meaning | The chosen value may be misleading |
| Missing indicator | Add a 0/1 column showing absence | Missingness itself may predict the target | Adds another feature |
| K-nearest neighbours | Use similar rows to estimate the value | Relationships between features are strong | More expensive and scale-sensitive |
| Model-based | Predict the missing value with another model | Large, structured datasets | Can overfit and is more complex |

## When is imputation needed?

Use imputation when the model cannot accept missing values, deleting incomplete rows would remove too much data, and a sensible replacement can be justified. Consider dropping a column when it is almost entirely missing and has little business value.

## Missing-data mechanisms

- **MCAR:** missingness is unrelated to observed or unobserved information.
- **MAR:** missingness can be explained by other observed columns.
- **MNAR:** the reason for missingness is related to the missing value itself.

These are questions for investigation, not labels to apply without understanding the data-collection process.

## Advantages

- Keeps more rows available for training.
- Makes a dataset compatible with more algorithms.
- Can be implemented consistently in a pipeline.
- A missing indicator can preserve useful information about absence.
- Median and mode rules are simple to explain and reproduce.

## Risks and limitations

- A single statistic reduces natural variation.
- Mean imputation is pulled by outliers.
- Mode imputation can create an artificial majority category.
- A bad constant can create a fake pattern.
- Advanced imputers can be slow, difficult to explain, or prone to leakage.
- Imputation does not repair a faulty measurement; it only fills a blank.

## Correct step-by-step workflow

1. Count missing values by column and by row.
2. Read the data dictionary and ask what a blank means.
3. Decide whether to keep, drop, or transform the field.
4. Split into training and test data.
5. Fit the imputer on training data only.
6. Transform training, validation, test, and future records with that fitted rule.
7. Add a missing indicator when absence could be informative.
8. Compare model performance and inspect the distribution after imputation.

Never calculate the median or mode from the complete dataset before the split. That exposes test-set information to training and causes data leakage.

## Feature-engine examples in this folder

- [addmissingIndicator.ipynb](addmissingIndicator.ipynb): missing indicators with Feature-engine.
- [assmissingindicator_scikit.ipynb](assmissingindicator_scikit.ipynb): a scikit-learn style comparison.
- [houseprice.csv](houseprice.csv): Ames house-price data used in the examples.

Typical Feature-engine components are AddMissingIndicator, MeanMedianImputer, and CategoricalImputer. Put them inside a scikit-learn Pipeline when evaluating a model.

## Applications

- **Healthcare:** missing lab measurements, where not tested may itself be important.
- **Banking:** incomplete income or employment information.
- **Retail:** missing product attributes or customer demographics.
- **IoT:** sensor readings lost during a connection outage.
- **House prices:** missing basement, garage, or frontage information.
- **Surveys:** unanswered questions and not-applicable responses.

## Interview answer

> I choose an imputation strategy from the variable type, distribution, and meaning of missingness. I fit it on the training fold only, optionally add a missing indicator, and validate that the model is not benefiting from leakage. Median imputation is often a robust baseline for skewed numerical data, while categorical fields commonly use a missing label or the mode.

## Quick checklist

- Is the blank a genuine absence or a collection error?
- Is the column numerical, categorical, or ordinal?
- Are outliers making the mean unreliable?
- Should missingness be represented by an indicator?
- Was the rule fitted only on training data?
- Does the transformed distribution still make business sense?
