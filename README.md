# Hourly Traffic Volume Forecasting at Road Junctions

Forecasting hourly traffic volumes at four road junctions using historical traffic data enriched with weather and special-event information, and comparing multiple forecasting approaches (XGBoost, ARIMA, LSTM).

## Problem Statement

Traffic congestion directly affects ride-sharing pricing (through time-based fares and surge pricing), commuter travel time, and urban mobility planning. This project builds a predictive model to forecast hourly vehicle counts at four road junctions, enabling better traffic management, driver repositioning, and congestion mitigation strategies.

## Data Sources

| Source | Details |
|---|---|
| **Traffic data** | 48,120 hourly records across 4 junctions, 1 Nov 2015 – 30 Jun 2017 (Junction 4 starts 1 Jan 2017) |
| **Weather data** | Hourly temperature, humidity, precipitation, rain, wind speed — via [Open-Meteo Historical Weather API](https://open-meteo.com/) |
| **Event data** | Public holidays (national + Karnataka state) via the `holidays` Python library, plus RCB/IPL home-match dates (Chinnaswamy Stadium, Bengaluru) |

## Approach

1. **Research** — studied how traffic congestion affects ride-sharing pricing and business operations (surge pricing, ETA accuracy, driver supply)
2. **Data integration** — merged traffic, weather, and event data on hourly timestamps, correcting a GMT/IST timezone mismatch in the raw weather pull
3. **EDA & feature engineering** — built time-based features (hour, day-of-week, weekend flag, month, rush-hour flag) and lag features (1hr, 24hr, 168hr, 3hr rolling average) to capture daily/weekly traffic cycles
4. **Peak-hour analysis** — identified per-junction peak hours, day-of-week and seasonal patterns, and the influence of rain/holidays/events on congestion
5. **Model development** — trained and compared three model families with a strict time-based train/validation split
6. **Evaluation & refinement** — assessed models with MAE, RMSE, and R², used time-series cross-validation, diagnosed residual bias, and refined the feature set

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **XGBoost — Refined (top features)** | **2.861** | **4.612** | **0.9713** |
| XGBoost — Tuned | 2.901 | 4.730 | 0.9698 |
| XGBoost — Baseline | 2.874 | 4.772 | 0.9693 |
| LSTM | 4.475 | 6.155 | 0.9326 |
| ARIMA(2,0,2) | 28.616 | 36.037 | -1.3128 |

The refined XGBoost model (trained on the top 12 most important features) was the best performer across every metric, explaining ~97% of the variance in hourly traffic volume with an average error of under 3 vehicles/hour.

## Key Findings

- **Junction 4 breaks the typical rush-hour pattern** — it peaks around midday (12:00–15:00) instead of the evening rush (19:00–21:00) seen at the other three junctions, an insight that only emerged from per-junction analysis.
- **ARIMA fails to capture daily seasonality** — its forecast flattens to the series mean within a day, producing a negative R² (worse than predicting the average). This demonstrates why a feature-rich model that explicitly encodes time-of-day and lag information outperforms a plain univariate time-series baseline here.
- **Holidays and special events showed *lower* peak-hour traffic than normal days** — a counter-intuitive finding suggesting that while events concentrate traffic locally, they reduce general commuter traffic city-wide.
- **The single strongest predictor is the previous hour's traffic** (`vehicles_lag_1h`), accounting for roughly two-thirds of feature importance — confirming traffic's strong autocorrelation.

## Repository Structure

```
traffic-volume-forecasting-uber/
├── README.md
├── data/
│   ├── Dataset_Uber_Traffic.csv
│   ├── Integrated_Traffic_Weather_Event_Dataset.csv
│   └── Traffic_Dataset_Preprocessed_FeatureEngineered_Yashas.csv
├── notebooks/
│   ├── 01_EDA_feature_engineering.ipynb
│   └── 02_model_development_evaluation.ipynb
├── reports/
│   ├── Report_on_Effect_of_Traffic_on_Uber_Business.pdf
│   ├── Comprehensive_Report_on_Peak_Hour_Traffic_Analysis.pdf
│   └── Report_on_Model_Evaluation_and_Refinement.pdf
└── charts/
    ├── hourly_avg_by_junction.png
    ├── heatmap_hour_dow.png
    ├── weekday_vs_weekend.png
    ├── monthly_seasonal.png
    ├── rain_impact_peak.png
    ├── event_impact.png
    ├── radial_traffic_clock.png
    ├── arima_forecast_vs_actual.png
    ├── gbm_diagnostics.png
    ├── gbm_feature_importance.png
    ├── lstm_training_history.png
    └── residual_bias_by_hour.png
    
```

## Tech Stack

**Python** · **Pandas** / **NumPy** · **XGBoost** · **statsmodels** (ARIMA) · **TensorFlow / Keras** (LSTM) · **scikit-learn** · **Matplotlib** / **Seaborn**

## Methodology Notes

- Train/validation split is **time-based** (last 20% of the timeline held out), not random — since lag features already encode recent history, a random split would leak future information into training.
- Hyperparameter tuning and cross-validation use **`TimeSeriesSplit`**, ensuring every validation fold is always chronologically after its training fold.
- Feature importance and residual analysis (by hour of day) were used to diagnose systematic model bias and guide feature refinement.
