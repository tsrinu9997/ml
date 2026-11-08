# Discretisation: turning continuous numbers into bins

## What is discretisation?

Discretisation, also called binning, converts a continuous numerical feature into a small number of intervals. Instead of using an exact age such as 43.7, a model may use the group 35–50. Instead of exact income, it may use low, medium, and high income bands.

The result is similar to replacing a detailed ruler with labelled sections. It can make a pattern easier for people to understand, but it also removes detail.

## Why do we need it?

Discretisation can help when:

- the business already uses bands or thresholds;
- the relationship between a feature and target is clearly non-linear;
- small measurement differences are mostly noise;
- a linear model needs a simple way to represent different ranges;
- explainability is more important than exact numerical precision.

It is not automatically better than keeping the original continuous feature. Compare both approaches with validation.

## Main methods

### Arbitrary or domain-based bins

The analyst supplies boundaries. For example:

    age 0-17       -> child
    age 18-64      -> adult
    age 65 and up  -> senior

This is best when a regulation, business policy, or subject-matter expert defines meaningful cut points.

### Equal-width bins

The numerical range is split into intervals of equal size. A range from 0 to 100 with five bins becomes 0–20, 20–40, and so on. The bins are easy to describe, but some may contain very few observations.

### Quantile or equal-frequency bins

Boundaries are selected so that bins contain roughly similar numbers of rows. This avoids empty bins but can create boundaries that are difficult to explain.

### Decision-tree discretisation

A small decision tree learns cut points that help predict the target. This can capture useful non-linear thresholds, but the cut points must be learned from training data only.

## Advantages

- Easy to explain to non-technical stakeholders.
- Can represent a step-like or non-linear relationship.
- Can reduce the influence of tiny numerical fluctuations.
- Can make dashboards, rules, and policies easier to build.
- Often works well with linear models when exact values are not essential.

## Limitations

- Values in one bin become indistinguishable.
- Poor boundaries can reduce predictive power.
- Too many bins can overfit.
- Boundaries can be unstable between samples.
- A binned feature may hide an important trend inside the interval.
- Decision-tree bins can leak target information if fit before splitting.

## Step-by-step workflow

1. Plot the continuous feature against the target.
2. Decide whether bins have a business meaning or should be learned.
3. Choose a small, defensible number of bins.
4. Fit boundaries using training data only.
5. Apply the fitted boundaries to validation, test, and future data.
6. Inspect row counts per bin and check for empty or tiny groups.
7. Compare the binned model with a continuous-feature baseline.

Current Feature-engine versions require DecisionTreeDiscretiser precision to be None or a positive integer. For example, precision=3 is valid; precision=0 raises a ValueError.

## Notebooks in this folder

- [arbitrarydiscretiser.ipynb](arbitrarydiscretiser.ipynb): domain-supplied interval boundaries.
- [arbitrarydiscretiser_plus_meanencoder.ipynb](arbitrarydiscretiser_plus_meanencoder.ipynb): bins followed by mean encoding.
- [decisionTreediscretiser.ipynb](decisionTreediscretiser.ipynb): decision-tree-based boundaries.

## Applications

- **Marketing:** age or spending bands for campaign rules.
- **Banking:** risk bands used in policy and approvals.
- **Healthcare:** clinically meaningful measurement ranges.
- **Manufacturing:** temperature or pressure safety zones.
- **House prices:** area and age bands for interpretable pricing reports.
- **Customer support:** response-time bands such as fast, normal, and delayed.

## Interview answer

> Discretisation converts a continuous variable into intervals. I use it when thresholds are meaningful or a non-linear relationship should be represented in an interpretable way. I learn boundaries on the training fold, inspect sample counts, and compare against the original continuous feature because binning sacrifices information.

## Common mistakes

- Choosing boundaries after looking at the test-set target.
- Creating too many bins.
- Using arbitrary boundaries without a business explanation.
- Forgetting to handle values outside the training range.
