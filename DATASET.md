# CKD Dataset Summary

## Source
This project uses the public Chronic Kidney Disease dataset from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease

## Dataset file
- `data/raw/ckd_dataset.csv`

## Structure
The raw file contains a non-standard header layout with an initial metadata row and a second row describing the variable type. The project code strips these rows before modeling.

## Observed dataset properties
- Rows: 200
- Columns: 29
- Target column: `class`
- Target classes: `ckd`, `notckd`
- Missing values: represented using `NaN` and some values like `?`, blank strings, and range-type entries such as `< 0` or `1.019 - 1.021`
- Categorical and mixed-type values are converted to clean numeric or categorical features before modeling without leaking test data into training.

## Notes
The dataset includes clinical laboratory variables such as blood pressure, specific gravity, albumin, serum creatinine, hemoglobin, blood glucose, and associated comorbid features like hypertension, diabetes, and anemia.
