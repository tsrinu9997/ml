# Date-Time Features

This folder adapts the date and time topics from the [referenced Section 15 GitHub examples](https://github.com/solegalli/feature-engineering-for-machine-learning/tree/main/Section-15-Date-Time-Features). A raw timestamp is usually not useful to a machine-learning model by itself. We extract meaningful pieces such as year, month, weekday, hour, elapsed duration, and local timezone.

## Why create date-time features?

A timestamp is a compact container of many signals:

- calendar position: year, quarter, month, week, and day;
- clock position: hour, minute, and second;
- business meaning: weekend, month-end, holiday, or working hours;
- elapsed time: days since signup, time between purchase and delivery;
- location context: local hour and weekday after timezone conversion.

For example, the timestamp 2026-11-08 18:30 contains a year, a season, a Sunday/weekend flag, an evening hour, and a position in a repeating daily cycle. Extracting these signals lets a model learn useful patterns.

## Notebooks

- [extract_date_features.ipynb](extract_date_features.ipynb): year, quarter, month, week, day, weekends, month boundaries, and calendar visualizations.
- [extract_time_features.ipynb](extract_time_features.ipynb): hour, minute, second, parts of day, and cyclical sine/cosine representations.
- [elapsed_time_features.ipynb](elapsed_time_features.ipynb): durations between events and time since a reference event.
- [timezone_features.ipynb](timezone_features.ipynb): UTC conversion, local times, daylight-saving changes, and safe comparisons.
- [datetime_features_feature_engine.ipynb](datetime_features_feature_engine.ipynb): automated extraction with Feature-engine inside a scikit-learn pipeline.

## Choosing the right feature

| Question | Useful feature |
|---|---|
| Does the trend change over years? | Year or time since a fixed reference |
| Does the pattern repeat each year? | Month, week, day of year, or seasonal phase |
| Does behaviour change by weekday? | Day of week and weekend flag |
| Does behaviour change by time of day? | Hour, minute, part of day, or cyclic hour |
| How long since an event? | Elapsed hours or days |
| Do users live in different regions? | Convert to a common reference and derive local features |

## Important modelling decisions

### Cyclical variables

Raw hour values make 23:00 appear far from 00:00 even though they are adjacent. For repeating patterns, use:

    sin_value = sin(2 * pi * value / period)
    cos_value = cos(2 * pi * value / period)

Use period 24 for hour, 7 for weekday, and 12 for month. Keep the original feature as well when it has a useful interpretation.

### Leakage and availability

Only create a feature from information available at prediction time. A delivery completion date cannot be used to predict delivery delay before the delivery happens. “Days since last event” must use the latest event known at the prediction timestamp.

### Timezones

Convert timestamps to a well-defined timezone before extracting local hour, weekday, or weekend. Two records with the same UTC instant may have different local dates and hours. Never compare naive timestamps from different regions as if they were in the same timezone.

### Train/test and time order

For forecasting or future prediction, split chronologically rather than randomly. Future information must never influence past training rows. For ordinary tabular prediction, still fit any learned date transformation within the training fold.

## Applications

- Retail demand by weekday, month, and hour.
- Fraud detection by local transaction time.
- Customer churn using days since signup or last activity.
- Delivery prediction using elapsed processing time.
- Energy forecasting using hour, season, and daylight-saving context.
- Healthcare operations using appointment hour and waiting duration.
- Web analytics using session start hour and time since the previous visit.

## Feature-engine API

Feature-engine provides DatetimeFeatures, which uses pandas datetime operations to extract features and can be placed in a scikit-learn pipeline. The notebook demonstrates features such as month, quarter, year, week, day of week, day of month, day of year, weekend, month boundaries, leap year, hour, minute, and second.

## Interview answer

> I do not feed a raw timestamp directly into most tabular models. I parse it with an explicit timezone policy, extract calendar and clock components, represent repeating cycles with sine and cosine when appropriate, and calculate elapsed durations only from information available at prediction time. For time-series problems I use a chronological split to avoid future leakage.

## Run the notebooks

From the project root:

    python -m pip install pandas numpy matplotlib seaborn scikit-learn feature-engine jupyter
    jupyter notebook

The examples create small reproducible dataframes, so no external dataset download is required.
