# AI_For_Market_Trend_Analysis

# 📈 AI Stock Market Prediction

This project leverages **Machine Learning** and **Time-Series Forecasting** to analyze and predict **Indian stock market prices**.  
It uses **2 years of stock data (Tata Motors, NSE India)** and builds both **predictive models** and **visual insights** to help traders and analysts.

---

## 🎯 Motivation

Stock markets are influenced by price action, momentum, volatility, and trading volume.  
This project aims to:

- Understand **past trends** using technical indicators such as
  - **SMA20 & SMA50** (trend direction and crossovers)  
  - **RSI (14)** (momentum strength, overbought/oversold levels)  
  - **MACD & Signal Line** (bullish/bearish momentum shifts) 
- Predict **future stock prices** with Facebook Prophet  
- Classify **market direction** (up or down) with ML models  
- Detect **anomalies** (sudden price/volume spikes) for risk management  

---



## 🚀 Features

- **Data Preprocessing**
  - Handle missing values  
  - Convert dates, sort chronologically  
  - Create **targets**: next-day price, return %, binary trend  

- **Feature Engineering**
  - Calculate **technical indicators**:  
    - **SMA (20, 50)** – simple moving averages for trend direction  
    - **RSI (14)** – momentum indicator (overbought/oversold)  
    - **MACD, Signal, Histogram** – momentum & reversals  
    - **Volatility (5-day rolling)** – short-term risk measure  
    - **ROC (Rate of Change)** – momentum strength  
    - **Relative Strength Index** – confirm buy/sell pressure  
  - **Lag features** for historical dependency

---

## 📊 Technical Indicator Thresholds

### 1️⃣ RSI (Relative Strength Index, 14)
- **Range**: 0 → 100  
- **Interpretation**: Measures momentum → how strong buying/selling pressure is.  

| RSI Value | Market Condition | Meaning |
|-----------|-----------------|---------|
| **> 70** | Overbought | Too many buyers → price may **reverse down** soon |
| **50 – 70** | Bullish | Strong buying pressure, healthy uptrend |
| **30 – 50** | Bearish | Selling pressure, mild downtrend |
| **< 30** | Oversold | Too many sellers → price may **reverse up** soon |

📌 *Example*: If Tata Motors RSI = **78** → stock is **overbought**, possible pullback.  

---

### 2️⃣ SMA 20 & SMA 50 (Simple Moving Averages)
- **SMA20** → Short-term trend (≈ 1 month of trading days)  
- **SMA50** → Medium-term trend (≈ 2.5 months of trading days)  

| SMA Relationship | Signal | Meaning |
|------------------|--------|---------|
| **SMA20 > SMA50** | Bullish Crossover | Short-term stronger → **uptrend** starting |
| **SMA20 < SMA50** | Bearish Crossover | Short-term weaker → **downtrend** |
| **Both flat / close together** | Neutral | Price is **sideways / consolidating** |

📌 *Example*: If SMA20 crosses above SMA50 → called a **Golden Cross** → bullish signal.  

---

### 3️⃣ MACD (Moving Average Convergence Divergence)
- **Components**:  
  - MACD Line = 12-day EMA – 26-day EMA  
  - Signal Line = 9-day EMA of MACD Line  
  - Histogram = MACD – Signal  

| Condition | Signal | Meaning |
|-----------|--------|---------|
| **MACD > Signal Line** | Bullish | Buyers gaining strength |
| **MACD < Signal Line** | Bearish | Sellers gaining strength |
| **MACD ≈ Signal (flat)** | Neutral | Weak trend / sideways |
| **MACD far above 0** | Strong Uptrend | Momentum bullish |
| **MACD far below 0** | Strong Downtrend | Momentum bearish |

📌 *Example*: If MACD line crosses above Signal line → **bullish crossover** → possible uptrend.  

---

## ✅ Quick Summary
- **RSI > 70** → Overbought (possible fall)  
- **RSI < 30** → Oversold (possible rise)  
- **SMA20 > SMA50** → Bullish (Golden Cross)  
- **SMA20 < SMA50** → Bearish (Death Cross)  
- **MACD > Signal** → Bullish, **MACD < Signal** → Bearish  



- **Modeling**
  - **Machine Learning**: Logistic Regression, XGBoost, LightGBM  
  - **Time-Series**: Facebook Prophet (daily & weekly predictions)  
  - Comparison of performance across models  

- **Evaluation Metrics**
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
  - Confusion Matrix  

- **Visualization**
  - Forecasts (daily & weekly) using Prophet + Plotly  
  - Technical indicator overlays (SMA, RSI, MACD)  
  - Volume vs Price relationships  
  - Anomaly detection (highlight unusual spikes)  

---

## 📊 Dataset

- **Stock**: Tata Motors (NSE India)  
- **Timeframe**: Last 2 years (~3000 rows)  
- **Columns**:  
  - `date`, `open`, `high`, `low`, `close`, `volume`  
  - Technical features: `sma_20`, `sma_50`, `rsi_14`, `macd`, `returns`, `volatility`, `roc_5`  
- **Source**: NSE India public data
- The raw dataset included columns such as:  
  - `Date`  
  - `Open`  
  - `High`  
  - `Low`  
  - `Close`  
  - `Volume`  
  - `VWAP`  

Unnecessary or redundant columns such as **Series** were ignored.  

A **correlation heatmap** was also used to identify and drop unrelated or highly collinear columns, ensuring only meaningful features were kept for model training and analysis.  

---

## Models Used  

Two models were applied for different objectives:  

### 1. Random Forest Classifier  
- **Input:** Technical indicators and derived features  
- **Output:** Predicts **Up (1)** or **Down (0)** trend for the next day  
- **Training:** 80% historical data, 20% testing using time-based split  
- **Evaluation:** Accuracy ~ **75%**, with **Precision, Recall, and F1-score** used for validation  

### 2. Prophet (by Facebook)  
- **Input:** `Date (ds)`, `Closing Price (y)`  
- **Output:** Forecasts stock prices for the **next 30 days**  
- **Accuracy (Evaluation Metrics):**  
  - Root Mean Squared Error (**RMSE**): `30.12`  
  - Mean Absolute Error (**MAE**): `23.22`  
- **Visualization:** Charts included trend with **upper and lower bounds** to evaluate accumulation zones.  
  Volume overlays were added to check if the stock is under **accumulation or distribution** phases.  

---

## Results & Evaluation  

- **Random Forest Classifier**  
  - Achieved around **75% classification accuracy**  
  - Effectively captured **SMA20, SMA50, RSI(14), MACD, and volatility** patterns  
  - Supported decisions about **bullish, neutral, or bearish momentum**  

- **Prophet (Time-Series Forecasting)**  
  - Produced reliable **30-day forecasts**  
  - Forecast plots displayed **upper and lower thresholds**, useful for identifying whether a stock was:  
    - **Consolidating (accumulation phase)**  
    - **Showing strong directional momentum**  
  - Adding **volume overlays** to Prophet charts enhanced detection of **accumulation/distribution phases**  

---

## Conclusion  

This project demonstrates how combining **correlation analysis**, **technical indicators**, and **machine learning** with **time-series forecasting** can provide **actionable insights** for traders and investors.  

- **Random Forest** → Assists in **daily buy/sell decisions** using technical signals  
- **Prophet** → Provides **broader forecasts** with uncertainty intervals  

Together, they offer a **comprehensive toolkit** for understanding **market momentum** and improving **risk management**.  

## 🛠️ Installation & Usage

### 1️⃣ Clone Repo
```bash
git clone https://github.com/yourusername/AI_stock_market.git
cd AI_stock_market
