# A Beginner’s Guide to Encoding Data

## What does “encoding” mean?

Encoding means changing information into a form that a computer model can use. A machine learning model works with numbers. But many useful facts are words or labels:

- a passenger boarded at **Southampton**
- a customer gave a **Good** review
- a ticket belongs to **First Class**

Encoding turns those categories into numbers or columns while trying to preserve the meaning that matters.

A simple way to think about it: encoding is like translating a form into a language the model can read. A good translation keeps the original meaning. A poor translation can accidentally suggest something that is not true.

## Why do we need encoding?

Most common machine learning algorithms perform calculations with numbers. They cannot directly calculate with words such as `female`, `male`, `Good`, or `Poor`. Encoding lets us use this information in those calculations.

Encoding also helps us represent different kinds of categories correctly:

- Some categories have a real order, such as `Poor < Average < Good`.
- Some categories are just different names, such as `Southampton`, `Cherbourg`, and `Queenstown`.

These two cases should usually be encoded differently. Giving arbitrary numbers to unordered names can mislead a model into thinking that one name is “more” than another.

## A small example

Suppose we have a column called `Port`:

| Passenger | Port |
|---|---|
| A | Southampton |
| B | Cherbourg |
| C | Southampton |

The computer needs a numeric representation. We could create one column for each port:

| Passenger | Port_Southampton | Port_Cherbourg | Port_Queenstown |
|---|---:|---:|---:|
| A | 1 | 0 | 0 |
| B | 0 | 1 | 0 |
| C | 1 | 0 | 0 |

A `1` means “this passenger belongs to this category”; `0` means “does not.” This is **one-hot encoding**.

## Common encoding methods

### 1. One-hot encoding

**What it does:** Makes a separate yes/no column for each category.

For `Color = [Red, Blue, Green]`:

| Color | Color_Red | Color_Blue | Color_Green |
|---|---:|---:|---:|
| Red | 1 | 0 | 0 |
| Blue | 0 | 1 | 0 |
| Green | 0 | 0 | 1 |

**Use it for:** Categories that have no natural order, such as port, city, color, or gender.

**Advantages:** It does not claim that one category is larger or better than another. The result is easy to understand.

**Trade-off:** A column with many unique categories creates many new columns. If a feature has 500 different cities, one-hot encoding could add 500 columns. Rare categories may also appear only a few times.

**Reference category:** Sometimes one dummy column is left out. For a two-category feature, instead of columns for both `Red` and `Blue`, one column such as `Color_Red` is enough: `1` means Red and `0` means Blue. This is useful for some linear models because it avoids redundant columns. The omitted category becomes the reference group. Keeping every column is also appropriate for many models.

**Unknown values:** A new category not seen during training needs a defined policy. Common options are to ignore it, map it to `Other`, or retrain with the new category. Do not assume every encoder handles new categories the same way.

### 2. Ordinal encoding

**What it does:** Replaces ordered categories with rank numbers.

For a review scale:

| Review | Code |
|---|---:|
| Poor | 0 |
| Average | 1 |
| Good | 2 |

**Use it for:** Categories where the order is real and meaningful, such as satisfaction (`Poor < Average < Good`) or education levels where the definitions support an order.

**Important caution:** The numbers show order, but can also make some models treat the gaps as equal. The difference between `Poor` and `Average` may not be the same in real life as the difference between `Average` and `Good`. Decide whether that assumption makes sense for your model.

Do not encode unordered categories like ports as `Southampton=1`, `Cherbourg=2`, `Queenstown=3`. That would introduce a made-up order and distance.

### 3. Label encoding

**What it does:** Gives each category an integer, for example `Red=0`, `Blue=1`, `Green=2`.

**Use it carefully:** It is commonly useful for encoding the **target labels** that a classifier is asked to predict (for example, `No=0`, `Yes=1`). For input features, integer labels can make an algorithm believe that categories have an order or numeric distance. For unordered input categories, one-hot encoding is often safer.

> “Label encoding” is sometimes used as a general phrase for turning categories into numbers. Check what a specific library’s encoder does and whether it is being applied to input features or the target.

### 4. Binary encoding

**What it does:** Assigns each category a number, then writes that number using binary digits (0s and 1s). Those digits become a small set of columns.

It can represent many categories using fewer columns than one-hot encoding. It is useful for high-cardinality features, but the columns are harder to explain and their binary patterns do not necessarily carry useful category relationships.

### 5. Frequency or count encoding

**What it does:** Replaces a category with how often it appears, or what fraction of rows have it.

For example, if Southampton appears in 100 rows, it may become `100` or `0.50`.

**Use it for:** Features with many categories, when category frequency may be informative and the model can work with this representation.

**Caution:** Two different categories with the same frequency become indistinguishable. Calculate counts from training data only and reuse them on test data; otherwise information from the test set can leak into training.

### 6. Target encoding

**What it does:** Replaces a category with a statistic of the outcome for that category. For example, a port could be represented by the average survival rate among passengers from that port.

**Why it can help:** It represents high-cardinality categories in a compact way.

**Major caution:** It can leak the answer into the features, especially for small datasets. Use a leakage-safe method such as cross-fitting or smoothing, and calculate encodings using training folds only. Do not calculate a category’s target average using the same row’s outcome and then train on that row.

### 7. Hash encoding

**What it does:** Uses a hash function to map category names into a fixed number of columns.

**Use it for:** Very large or continually changing sets of categories, such as words or product identifiers.

**Trade-off:** Different categories can land in the same column (a collision), and the resulting columns are not as easy to interpret.

### 8. Date and time features

Dates are not ordinary categories, but often need conversion into useful numeric information. From a date, we might extract year, month, weekday, or hour. For repeating cycles such as hour of day or month of year, sine/cosine features can preserve the fact that the end and beginning of the cycle are close (11 PM is near midnight).

## Which method should I choose?

| Kind of information | Example | Good starting choice |
|---|---|---|
| Unordered categories with a few values | Port, color | One-hot encoding |
| Categories with a meaningful order | Review: Poor, Average, Good | Ordinal encoding |
| Class label the model predicts | Purchased: No / Yes | Keep as target; map to 0/1 if needed |
| Many categories | City, product code | Frequency, binary, hashing, or carefully validated target encoding |
| Date/time | Purchase date | Extract meaningful calendar features; use cyclic features for repeating cycles |

The best choice also depends on the model. Linear models, tree models, and neural networks can respond differently to the same representation. When building a predictive model, compare sensible alternatives using validation data.

## Encoding on the Titanic data

The Titanic files in this folder provide useful examples:

- `sex` is unordered. Use one-hot encoding rather than assigning female and male arbitrary ranks.
- `embarked` (port) is unordered. One-hot encoding is a natural starting point.
- `pclass` has class levels 1, 2, and 3. It can be treated as an ordered category if that matches the question, or as a set of separate categories if you do not want to assume a rank relationship.
- `survived` is the outcome, not an input feature. Keep it separate from the columns used to predict survival.
- `boat` and `body` describe information available after the event. Using them to predict survival would reveal information that would not be available beforehand (target leakage).
- `name`, `ticket`, and `cabin` can have many values. They may contain useful information, but need deliberate feature engineering (such as extracting a title, ticket group, or cabin deck) before encoding.

## A safe workflow

1. Decide whether each column is a measurement, an ordered category, an unordered category, an identifier, or the target.
2. Inspect category values, spelling differences, rare values, and missing values.
3. Choose an encoding that matches the meaning. Do not invent an order.
4. Split data into training and test sets before learning category frequencies or target statistics.
5. Fit the encoder on the training data only.
6. Use that same fitted encoder to transform test data and future observations.
7. Put encoders inside a machine learning pipeline when doing cross-validation. This helps prevent information from crossing between training and validation folds.
8. Check the transformed columns and sample rows so you know what the codes mean.

## Encoding is not scaling

Encoding represents categories. **Scaling** changes the numeric range of measurements such as age or fare. For example, min–max scaling might map ages into the range 0–1. You may encode categorical columns and scale numeric columns in the same preprocessing pipeline, but they solve different problems.

## Final takeaway

Choose an encoding that preserves what a category means:

- Use **one-hot encoding** when labels are different names with no ranking.
- Use **ordinal encoding** when the order is meaningful and you are comfortable with the assumptions the numbers introduce.
- Use other encodings for many-category features only when their trade-offs and leakage risks are understood.

When uncertain, start with the simplest representation that matches the data, inspect it, and compare alternatives using a proper validation workflow.
