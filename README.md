# Multi-variate Time Series Forecasting - Air Quality

## Project Overview
This project focuses on multi-variate time series forecasting applied to air quality prediction. Using a large-scale dataset from the UCI Machine Learning Repository, we investigated various forecasting techniques including VAR, SARIMAX, FB Prophet, LSTM (with a focus on stacked bidirectional architecture), Ensemble approaches, TCN, and GRU models. Each model was fine-tuned for optimal performance, and their accuracy was evaluated using the Root Mean Squared Error (RMSE) metric.

## Table of Contents
1. [Introduction](#introduction)
2. [Methods](#methods)
   - [Data Preprocessing](#data-preprocessing)
   - [Exploratory Data Analysis](#exploratory-data-analysis)
   - [Modeling](#modeling)
   - [Testing](#testing)
   - [Evaluation](#evaluation)
   - [Time Series Forecasting](#time-series-forecasting)
3. [Experimental Setup](#experimental-setup)
   - [VAR Model](#var-model)
   - [SARIMAX](#sarimax)
   - [FB Prophet](#fb-prophet)
   - [LSTM Models](#lstm-models)
   - [Ensemble Models](#ensemble-models)
   - [Temporal Convolutional Networks (TCN)](#temporal-convolutional-networks-tcn)
   - [Gated Recurrent Units (GRU)](#gated-recurrent-units-gru)
4. [Results](#results)
5. [Conclusion](#conclusion)
   - [Key Findings](#key-findings)
   - [Limitations](#limitations)
   - [Future Scope](#future-scope)
6. [References](#references)

## Introduction
Air quality poses a significant risk to public health and the environment. Accurate forecasting of air quality indices (AQI) is crucial for effective environmental monitoring and pollution control. This project systematically evaluates various machine learning and deep learning models to predict air quality indices, providing insights into their predictive capabilities and real-world applications.

## Methods
### Data Preprocessing
- Cleaned the dataset by removing irrelevant columns and NaN values.
- Standardized numerical values and merged 'Date' and 'Time' columns.
- Imputed missing data and corrected outliers to ensure data integrity.

### Exploratory Data Analysis
- Conducted visual analyses to compare pollution levels across different time periods.
- Analyzed relationships between temperature, humidity, and various pollutants.

### Modeling
- Used statistical and machine learning models including VAR, SARIMAX, FB Prophet, LSTM, TCN, and GRU.
- Performed hyperparameter tuning for each model to optimize performance.
- Evaluated models using RMSE and plotted forecasting graphs.

### Testing
- Validated models on test sets to assess their performance on new data.

### Evaluation
- Used RMSE as the primary evaluation metric to compare model accuracy.

### Time Series Forecasting
- Plotted forecasting graphs to visualize the predictive power of each model.

## Experimental Setup
### VAR Model
- Conducted stationarity tests and Granger causality tests.
- Selected the best lag order and evaluated the model using RMSE and MAE.

### SARIMAX
- Handled seasonal effects and selected hyperparameters using ACF and PACF plots.
- Evaluated the model with RMSE and plotted 30-day forecasts.

### FB Prophet
- Tuned hyperparameters and extended the model for multivariate analysis.
- Evaluated the model using RMSE and plotted forecasting graphs.

### LSTM Models
- Experimented with various LSTM configurations including Vanilla, Stacked, Bidirectional, and CNN-LSTM.
- Selected the best model through hyperparameter tuning and evaluated it using MAE and forecasting graphs.

### Ensemble Models
- Used Random Forest, Gradient Boosting, AdaBoost, Histogram GB, and XGBoost.
- Tuned hyperparameters using Randomized Search CV and evaluated models using RMSE and other metrics.

### Temporal Convolutional Networks (TCN)
- Performed hyperparameter tuning and evaluated the model using RMSE.
- Plotted forecasting graphs to visualize predictions.

### Gated Recurrent Units (GRU)
- Tuned hyperparameters using grid search and evaluated the model using RMSE.
- Plotted forecasting graphs to visualize predictions.

## Results
- The Stacked Bidirectional LSTM model achieved the lowest RMSE of 0.424, indicating superior prediction accuracy.
- The SARIMAX model performed well with an RMSE of 1.4, effectively handling seasonal patterns.
- GRU and TCN models showed competitive performance with RMSEs of 7.43 and 10.47, respectively.
- Ensemble methods, particularly XGBoost, required more tuning for optimal performance.

## Conclusion
### Key Findings
- The Stacked Bidirectional LSTM model was the most effective for air quality forecasting.
- Hyperparameter tuning significantly improved model accuracy.
- The graphical representation of forecasts provided valuable insights into model performance.

### Limitations
- Model performance is highly dependent on data quality and quantity.
- Computational intensity may limit the application of complex models.

### Future Scope
- Further refine SARIMAX for better seasonal accuracy.
- Improve Prophet model by adding more datasets and adjusting hyperparameters.
- Develop adaptive GRU architectures for handling dynamic data.

## References
1. Vito, Saverio. (2016). Air quality. UCI Machine Learning Repository. https://doi.org/10.24432/C5060Z
2. Beijing Multi-Site Air-Quality Data. https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
3. Time series analysis and forecasting of air quality index. https://doi.org/10.1063/5.0045753
4. U. A. Bhatti et al., "Time Series Analysis and Forecasting of Air Pollution Particulate Matter (PM2.5): An SARIMA and Factor Analysis Approach," in IEEE Access, vol. 9, pp. 41019-41031, 2021, doi: 10.1109/ACCESS.2021.3060744. https://ieeexplore.ieee.org/abstract/document/9359734
5. Brian S. Freeman, Graham Taylor, Bahram Gharabaghi Jesse Thé (2018) Forecasting air quality time series using deep learning, Journal of the Air Waste Management Association, 68:8, 866-886, DOI: 10.1080/10962247.2018.1459956. https://www.tandfonline.com/doi/full/10.1080/10962247.2018.1459956
