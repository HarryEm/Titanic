# Kaggle Titanic

This repo looks at two different approaches to Kaggle's famous [Titanic](https://www.kaggle.com/c/titanic) competition

## Notebooks

The files relate to two Kaggle notebooks exploring both how [feature engineering](https://www.kaggle.com/harryem/feature-engineering-on-the-titanic-for-0-81339) and two different approaches for dealing with [missing data](https://www.kaggle.com/harryem/titanic-comparing-two-approaches-for-missing-data) can improve predictive accuracy.

### Feature Engineering

The code found in

```
FeatureEngineering.Rmd
```

contains an R Markdown file containing data visualisation, cleaning, and various modelling approaches (logistic regression, lasso regularised logistic regression, random forrest and GBM).

### Missing Data

The code found in

```
MissingData.Rmd
```

contains an R Markdown file where I compare imputating missing data, to partitioning the training set according to which features (if any) are missing, and fitting separate random forrest classifiers for each.

### Data and final predictions

The train and test data can be found at

```
TitanicTrain.csv
TitanicTest.csv
```
And the final predictions with the best accuracy are in

```
predictions.csv
```
