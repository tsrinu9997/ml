# Feature Engineering: a beginner-friendly and interview-ready guide

Feature engineering means turning raw data into useful information for a machine-learning model.

Imagine that a model is a student taking an exam. Raw data is the textbook: it may contain missing answers, different units, words, unusual values, and numbers that are difficult to compare. Feature engineering cleans and translates that textbook into a form the student can use.

This folder is a practical collection of notebooks using [Feature-engine](https://feature-engine.trainindata.com/) and scikit-learn. Every notebook follows the same general idea:

1. Understand the data and the business meaning of each column.
2. Decide which column is the target to predict.
3. Split the data into training and test sets.
4. Learn preprocessing rules from the training data only.
5. Apply the same rules to validation, test, and future data.
6. Visualise what changed and evaluate the model.

The “training data only” rule is extremely important. If information from the test set is used to calculate a median, scaling range, category frequency, or target average, the model has received a sneak preview of the answers. This is called **data leakage**.

## Folder map

| Folder | Plain-language purpose | Main techniques | Example data |
|---|---|---|---|
| [`imputation/`](imputation/) | Fill in blank values | Median, categorical value, missing indicators | House prices |
| [`outliers/`](outliers/) | Handle unusually large or small observations | Winsorising, arbitrary capping, trimming | Titanic and house prices |
| [`normalization/`](normalization/) | Put numbers into a fixed range, usually 0–1 | Min–max normalization | Social Network Ads and wine |
| [`standardization/`](standardization/) | Express numbers as distance from their average | Z-score standardization | Social Network Ads |
| [`encoding/`](encoding/) | Translate words/categories into numbers | One-hot, ordinal, mean/target encoding | Titanic, cars, customers |
| [`discretisation/`](discretisation/) | Turn a continuous number into understandable bins | Arbitrary and decision-tree bins | House-price style data |
| [`pipelines/`](pipelines/) | Put the complete recipe into one repeatable object | Feature-engine + scikit-learn pipelines | Adult income, wine, houses |
| [`Date-Time-Features/`](Date-Time-Features/) | Turn timestamps into calendar, clock, duration, and local-time features | Pandas datetime operations and Feature-engine DatetimeFeatures | Synthetic order and event data |
| [`featurecreation/`](featurecreation/) | Combine columns and model non-linear relationships | Mathematical, relative, polynomial, and spline features | Ames house prices |

## A safe project workflow

### Step 1: Understand the columns

For every column, ask:

- Is it the target, an input, an identifier, or information that would only be known after the event?
- Is it numerical, categorical, ordinal, a date, or free text?
- What do missing values mean?
- Are the units comparable to other numerical columns?
- Could a very high or low value be a genuine observation or a data-entry mistake?

For example, in Titanic data, `survived` is the target. `boat` and `body` describe information after the rescue or death and should not be used when predicting survival before the voyage. They would cause target leakage.

### Step 2: Split before learning preprocessing rules

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

The test set is an exam paper. Keep it untouched until the end. A pipeline is the safest way to enforce this rule during cross-validation.

### Step 3: Fit transformations on training data only

```python
transformer.fit(X_train, y_train)      # learn medians, categories, limits, etc.
X_train_new = transformer.transform(X_train)
X_test_new = transformer.transform(X_test)
```

Never calculate a median, minimum, maximum, category frequency, or target mean using the complete dataset before the split.

### Step 4: Check the result

After every important transformation, inspect:

- shape before and after;
- missing values remaining;
- a few transformed rows;
- distributions and plots;
- whether train and test have compatible columns;
- model metrics on unseen data.

---

## 1. Imputation: what to do with missing values

### The layman explanation

Suppose a form has a blank “age” field. A model cannot do arithmetic with a blank cell. **Imputation** fills the blank with a sensible value. It is like replacing a missing answer with a carefully chosen estimate while keeping a note that the original answer was missing.

### Common choices

| Situation | Common solution | Reason |
|---|---|---|
| Numerical data with extreme values | Median | Less affected by unusually high or low values |
| Numerical data with a symmetric distribution | Mean | Uses all values, but is sensitive to outliers |
| Categorical data | Most frequent category or `Missing` | Keeps the column usable and makes the rule explicit |
| Missingness itself may carry meaning | Missing indicator | Adds a 0/1 column saying “this was missing” |
| A field is mostly blank and not useful | Drop the field | Avoids inventing too much information |

The notebooks in [`imputation/`](imputation/) show `AddMissingIndicator` and scikit-learn alternatives using the house-price data:

- [`addmissingIndicator.ipynb`](imputation/addmissingIndicator.ipynb)
- [`assmissingindicator_scikit.ipynb`](imputation/assmissingindicator_scikit.ipynb)
- [`houseprice.csv`](imputation/houseprice.csv)

### Step by step

1. Count missing values per column.
2. Understand why values are missing: random absence, “not applicable”, or a data-collection problem.
3. Split into train and test.
4. Fit the imputer on training data.
5. Transform both sets with the fitted imputer.
6. Confirm that no unexpected blanks remain.

### Interview answer

> I choose the imputation rule from the variable’s meaning and distribution. I fit it on the training fold only, and I add a missingness indicator when the fact that a value is absent may itself be predictive. Median imputation is often safer than mean imputation when outliers exist.

### Common mistake

Do not fill missing values using the full dataset before splitting. That lets test-set information influence the training process.

---

## 2. Outliers: unusually large or small observations

### The layman explanation

An outlier is a value far away from the usual pattern—for example, a house that is ten times larger than most houses, or a fare entered with an extra zero. An outlier may be a mistake, a rare but real case, or evidence that the process has more than one group.

An outlier is not automatically bad. First ask whether it is plausible and relevant.

### Three approaches in this project

| Method | What it does | When to choose it |
|---|---|---|
| `Winsorizer` | Caps extreme values at learned statistical limits; keeps all rows | The observation is real but should not dominate the model |
| `ArbitraryOutlierCapper` | Uses limits supplied by a domain expert or specification | A business, physical, or measurement boundary is known |
| `OutlierTrimmer` | Removes rows beyond the chosen limits | The observations are errors or outside the population of interest |

Examples:

- [`Winsorizer.ipynb`](outliers/Winsorizer.ipynb)
- [`ArbitraryOutlierCapper.ipynb`](outliers/ArbitraryOutlierCapper.ipynb)
- [`OutlierTrimmer.ipynb`](outliers/OutlierTrimmer.ipynb)
- Titanic versions: [`Winsorizer_titanic.ipynb`](outliers/Winsorizer_titanic.ipynb), [`ArbitraryOutlierCapper_titanic.ipynb`](outliers/ArbitraryOutlierCapper_titanic.ipynb), [`OutlierTrimmer_titanic.ipynb`](outliers/OutlierTrimmer_titanic.ipynb)

### How outliers are commonly detected

- **IQR rule:** values below `Q1 - 1.5 × IQR` or above `Q3 + 1.5 × IQR`.
- **Z-score rule:** values several standard deviations from the mean.
- **Quantile rule:** cap or flag a chosen percentile, such as the 99th percentile.
- **Domain rule:** a value outside a physically or commercially possible range.

### Step by step

1. Plot a histogram and boxplot.
2. Check the original record and unit of measurement.
3. Decide whether the value is an error, a valid rare case, or a different population.
4. Fit the capping/trimming rule on training data only.
5. Compare distributions and model performance before and after treatment.

### Interview answer

> I do not delete an outlier just because it is statistically unusual. I first check whether it is a valid observation. If it is valid but influential, I may cap it; if it is an impossible data-entry error, I correct or remove it; if it represents an important segment, I may create a feature or use a robust model instead.

---

## 3. Normalization: put values into a fixed range

### The layman explanation

Imagine comparing height measured in centimetres with salary measured in dollars. Salary has much larger numerical values, so some algorithms may pay it more attention simply because of its scale. **Normalization** resizes a feature, commonly to the range 0 to 1.

The min–max formula is:

\[
x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}
\]

The smallest training value becomes 0 and the largest becomes 1.

Notebook examples:

- [`normalization.ipynb`](normalization/normalization.ipynb) with [`Social_Network_Ads.csv`](normalization/Social_Network_Ads.csv)
- [`normalization_winedata.ipynb`](normalization/normalization_winedata.ipynb) with [`wine_data.csv`](normalization/wine_data.csv)

### When it helps

Normalization is useful for distance- and gradient-based algorithms such as k-nearest neighbours, k-means, neural networks, and some optimisation-based models. Tree models usually do not need scaling because they split by thresholds.

### Strength and weakness

- Strength: all features share a predictable range.
- Weakness: minimum and maximum are sensitive to outliers. A single extreme value can squeeze most observations into a tiny interval.

### Interview answer

> Min–max normalization maps a feature to a fixed interval, often [0, 1]. I fit the minimum and maximum on the training data only. I prefer it when a bounded range is useful and extreme values are controlled.

---

## 4. Standardization: measure distance from the average

### The layman explanation

**Standardization** changes a feature so that its training mean is approximately 0 and its training standard deviation is approximately 1. Instead of saying “this house has 2,000 square feet”, it says “this house is about 0.4 standard deviations above the average size”.

The z-score formula is:

\[
z = \frac{x - \mu}{\sigma}
\]

where `μ` is the training mean and `σ` is the training standard deviation.

Notebook examples:

- [`standaardization.ipynb`](standardization/standaardization.ipynb)
- [`standardization1.ipynb`](standardization/standardization1.ipynb)
- [`Social_Network_Ads.csv`](standardization/Social_Network_Ads.csv)

### Normalization versus standardization

| Question | Normalization | Standardization |
|---|---|---|
| Typical output | 0 to 1 | Mean 0, standard deviation 1 |
| Uses | Minimum and maximum | Mean and standard deviation |
| Sensitive to outliers | Often strongly | Also sensitive, but in a different way |
| Common use | Bounded inputs, distance methods | Linear models, SVMs, PCA, gradient methods |

Neither method is universally “better”. Choose based on the algorithm, data distribution, and business interpretation.

---

## 5. Encoding: translate categories into numbers

### The layman explanation

Models calculate with numbers, but data often contains words such as `male`, `Southampton`, `Good`, or `SUV`. **Encoding** translates those categories into a numerical representation without inventing a meaning that is not present.

The detailed encoding guide is [`encoding/readme.md`](encoding/readme.md). The working notebooks include:

- [`onehotencoding_titanic.ipynb`](encoding/onehotencoding_titanic.ipynb)
- [`onehotencoding_cars.ipynb`](encoding/onehotencoding_cars.ipynb)
- [`ordnialencodng_titanic.ipynb`](encoding/ordnialencodng_titanic.ipynb)
- [`ordinalencoding_cars.ipynb`](encoding/ordinalencoding_cars.ipynb)
- [`meanencoder_titanic.ipynb`](encoding/meanencoder_titanic.ipynb)
- [`ordinaryencoding.ipynb`](encoding/ordinaryencoding.ipynb)

### Which encoder should I use?

| Category type | Example | Good starting choice |
|---|---|---|
| Unordered labels | Port, colour, car brand | One-hot encoding |
| Meaningful order | Poor < Average < Good | Ordinal encoding |
| Target label | No / Yes | Keep separate as `y`; map to 0/1 if required |
| Many categories | City, product ID | Frequency, binary, hashing, or carefully cross-fitted target encoding |

### One-hot encoding

Creates one 0/1 column per category. It is easy to explain and does not create a fake order.

```text
Colour     Colour_Red   Colour_Blue   Colour_Green
Red             1            0             0
Blue            0            1             0
```

### Ordinal encoding

Maps ordered categories to ranks such as `Poor=0`, `Average=1`, and `Good=2`. Use it only when the order is real. A model may still assume the distance from 0 to 1 is comparable to the distance from 1 to 2.

### Target encoding warning

Target encoding uses the outcome to represent a category. It can be powerful for high-cardinality variables, but it is a major leakage risk. Use cross-fitting or a pipeline so each validation row is encoded without using its own target.

### Interview answer

> I select the encoder based on whether the categories are ordered, unordered, or high-cardinality. I fit it on the training fold only, define a policy for unseen categories, and check the transformed feature names. I never ordinal-encode an unordered feature merely because a library accepts integers.

---

## 6. Discretisation: turn a number into bins

### The layman explanation

Discretisation changes a continuous number into groups. Instead of using exact age, we might use `young`, `middle-aged`, and `older`. This can make a relationship easier to explain and can help a model capture a non-linear pattern.

Examples in [`discretisation/`](discretisation/):

- [`arbitrarydiscretiser.ipynb`](discretisation/arbitrarydiscretiser.ipynb)
- [`arbitrarydiscretiser_plus_meanencoder.ipynb`](discretisation/arbitrarydiscretiser_plus_meanencoder.ipynb)
- [`decisionTreediscretiser.ipynb`](discretisation/decisionTreediscretiser.ipynb)

### Two approaches

1. **Arbitrary or domain bins:** the analyst supplies boundaries, such as age 0–18, 19–65, and 66+.
2. **Decision-tree bins:** a small decision tree learns cut points that help predict the target.

### Trade-offs

- Bins are easy to explain and can reduce the influence of noisy small changes.
- Bins lose detail: two values in the same bin become identical.
- Too many bins can overfit.
- Tree-based boundaries must be learned inside the training fold to avoid leakage.

### Important error to remember

`DecisionTreeDiscretiser` requires `precision=None` or a positive integer. `precision=0` raises a `ValueError`; use a value such as `precision=3`, or omit the argument.

### Interview answer

> I discretise when the business meaning is naturally banded or when a non-linear relationship is easier to model with intervals. I choose boundaries from domain knowledge or training data only, and I compare the binned model with a model that keeps the original continuous feature.

---

## 7. Pipelines: make the recipe repeatable

### The layman explanation

A pipeline is an assembly line. Every new row passes through the same stations in the same order: clean it, create features, encode it, scale it, and make a prediction. The fitted stations remember the rules learned from the training data.

Pipeline notebooks:

- [`adult-income-with-feature-engine.ipynb`](pipelines/adult-income-with-feature-engine.ipynb): classification workflow for income prediction.
- [`create-new-features-with-feature-engine.ipynb`](pipelines/create-new-features-with-feature-engine.ipynb): creates domain-based wine features inside a model pipeline.
- [`predict-house-price-with-feature-engine.ipynb`](pipelines/predict-house-price-with-feature-engine.ipynb): imputation, encoding, discretisation, scaling, and Lasso regression for house prices.

### Basic pattern

```python
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('imputer', MeanMedianImputer()),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ('scaler', StandardScaler()),
    ('regressor', Lasso())
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Why pipelines matter

- They prevent preprocessing from accidentally seeing validation data.
- They keep training and prediction transformations identical.
- They make cross-validation reliable.
- They allow the full recipe to be saved and deployed.
- They make experiments easier to compare.

### Interview answer

> I put every learned preprocessing step and the estimator in one pipeline. During cross-validation, each fold fits the complete pipeline from scratch. This avoids leakage and guarantees that production data receives the same transformations as training data.

---

## How to choose a preprocessing sequence

There is no single mandatory order for every project, but this is a useful starting checklist:

1. Remove identifiers and post-outcome columns.
2. Split into train and test.
3. Create deterministic domain features such as age or totals.
4. Add missing indicators if absence may be informative.
5. Impute missing values.
6. Group rare categories.
7. Encode categorical variables.
8. Treat outliers if the model and domain require it.
9. Discretise or transform skewed numerical variables when justified.
10. Scale numerical features for algorithms that depend on distance or regularisation.
11. Fit the model and evaluate on unseen data.

The exact order can change. For example, an outlier rule should normally be learned after the train/test split, and a target encoder must be fitted in a leakage-safe cross-validation scheme.

## Interview quick reference

### What is feature engineering?

Feature engineering is the process of transforming raw columns into representations that make the target relationship easier for a machine-learning algorithm to learn.

### Why do we need it?

Real data contains blanks, words, different scales, outliers, dates, IDs, and non-linear relationships. Algorithms cannot automatically understand every business meaning.

### What is data leakage?

Data leakage happens when information that would not be available at prediction time—or information from the validation/test answers—enters the training features. It makes offline scores look better than real-world performance.

### When is scaling unnecessary?

Most tree-based models split by thresholds and are usually insensitive to feature scale. Linear models with regularisation, k-nearest neighbours, k-means, PCA, SVMs, and neural networks commonly benefit from scaling.

### How do you handle an unseen category?

Choose the policy deliberately: ignore it in one-hot encoding, map it to `Other`, use a fallback value, or retrain. The policy must be defined before deployment.

### How do you know whether a transformation helped?

Compare a sensible baseline and the transformed model with the same train/validation splits or cross-validation. Use metrics appropriate to the problem and inspect error patterns, not only one headline score.

## Running the notebooks

From the project root:

```bash
cd /Users/srinivasaraotadela/Documents/ml
python -m pip install pandas numpy matplotlib seaborn scikit-learn feature-engine jupyter
jupyter notebook
```

Then open a notebook, run the cells from top to bottom, and read the markdown before each code cell. If your environment is different, update the data paths in the loading cell rather than hard-coding a path that exists only on one computer.

## Final study advice

For every transformation, be able to answer four questions in an interview:

1. **What problem does it solve?**
2. **What assumption does it make?**
3. **What can go wrong, especially leakage?**
4. **How did you validate that it improved the model?**

If you can explain those four points in plain language, you understand the technique rather than merely memorising its class name.
