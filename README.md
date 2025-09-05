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

## Charts
## 1. Stock price with SMA
  <p align="center">
        <img width="1354" height="389" alt="image" src="https://github.com/user-attachments/assets/c8a151c7-96a0-4320-8965-fedd6b781df6" 
         style="border: 2px solid black; border-radius: 8px;"/>
  </p>

## Understanding the SMA (Simple Moving Average) Analysis  

The chart shows **stock price movements** along with three important trend indicators:  
- **SMA20 (blue line)** → Short-term trend (about 1 month)  
- **SMA50 (orange line)** → Medium-term trend (about 2-3 months)  
- **SMA200 (purple line)** → Long-term trend (almost a year)  

Candlesticks (red/green bars) show the actual daily price.  
By comparing price vs SMA lines, we can understand market momentum.  



### 🔎 What the Analysis Tells Us  

- **Swing Trading (8-10 weeks):**  
  - 📉 *Bearish Swing*: In the last 10 weeks, **SMA20 was below SMA50 ~66% of the time**.  
  - This suggests weakening momentum → swing traders should be cautious or look for short (sell) opportunities.  

- **Short-Term Trading (3-4 weeks):**  
  - 📉 *Bearish Short-Term*: In the last 4 weeks, prices closed **below SMA20 ~75% of the time**.  
  - This shows **selling pressure is strong**, meaning the short-term market trend is negative.  

- **Long-Term Investing (40+ weeks):**  
  - ⚠️ *Caution – Long-Term Bearish*: Over 40+ weeks, **SMA50 stayed below SMA200 ~98.5% of the time**.  
  - This is known as a **Death Cross** → signals long-term weakness and possible extended downtrend.  



### 🧠 What This Means for a Non-Stock Person  

- If you are a **short-term trader**, the signals say:  
  "Market is weak right now, better to be cautious or avoid aggressive buying."  

- If you are a **swing trader**, the signals say:  
  "Momentum is bearish → better opportunities may come if prices fall further and stabilize."  

- If you are a **long-term investor**, the signals say:  
  "The stock has been weak for months. Entering now may carry risk unless there is a clear recovery signal."  



👉 **In simple words:**  
The stock is showing **weakness in short, medium, and long-term views**.  
It may not be the best time to buy aggressively. Wait for signals of recovery (e.g., SMA20 crossing back above SMA50, or SMA50 moving above SMA200).  


## 2. RSI 14
<p align="center">
<img width="1354" height="352" alt="image" src="https://github.com/user-attachments/assets/9d9b5de4-d287-435d-bbc6-9ce5f2ad0b6e"
</p>
  
## Understanding RSI (Relative Strength Index)  

The RSI (14) chart shows momentum in the stock:  
- **Above 70** → Overbought (stock may be too expensive, risk of pullback)  
- **Below 30** → Oversold (stock may be too cheap, possible rebound)  
- **Between 30–70** → Neutral zone (no strong buying/selling pressure)  

In the chart:  
- The **red dashed line (70)** marks the overbought level.  
- The **green dashed line (30)** marks the oversold level.  
- The purple line shows RSI values over time.  



### 🔎 Trading Strategy Insights  

- **Short-Term Trading (5-6 weeks):**  
  - 🔵 *Neutral (RSI = 48.09)*  
  - The 4-week RSI average (43.84) suggests no strong buying or selling pressure.  
  - Traders should watch for moves closer to **30 (buy signal)** or **70 (sell signal)**.  

- **Swing Trading (~6 months):**  
  - 🟠 *Bearish Momentum (RSI = 39.85)*  
  - Average RSI over 2 weeks indicates selling pressure dominates.  
  - Swing traders should be cautious; market momentum is weak.  

- **Long-Term Investing (1+ year):**  
  - ⚖️ *Neutral Long-Term (RSI = 43.41)*  
  - The 1-year average RSI is in the **normal 30–70 range**.  
  - No strong long-term buy/sell signal → investors may wait for a clearer trend before entering.  



### 🧠 What This Means for a Non-Stock Person  

- **Short-term:** Market is currently stable, no strong buy/sell signals.  
- **Medium-term (swing trading):** Momentum is bearish → better to be cautious.  
- **Long-term investing:** The stock is neither overbought nor oversold. Investors should wait before making big moves.  

👉 **In simple words:**  
The RSI shows **weak momentum**. There’s no urgent buy or sell signal, but swing traders should be careful as the stock leans bearish.  


## 3. MACD


<p align="center">
  <img width="1354" height="365" alt="image" src="https://github.com/user-attachments/assets/04dc649d-e8be-4a44-8c5d-c4585ccaa975"

</p>

## Understanding MACD (Moving Average Convergence Divergence)  

The MACD helps measure **momentum** and shows when a stock might be shifting between **bullish (uptrend)** and **bearish (downtrend)** phases.  

In the chart:  
- **Blue Line = MACD Line**  
- **Orange Line = Signal Line**  
- **Gray Bars = Histogram (difference between MACD & Signal)**  

📌 Key rules:  
- If **MACD > Signal Line** → Bullish momentum (buyers stronger)  
- If **MACD < Signal Line** → Bearish momentum (sellers stronger)  
- If **MACD > 0** → Positive momentum  
- If **MACD < 0** → Negative momentum  



### 🔎 Trading Strategy Insights  

- 📉 **Bearish Bias:**  
  - In the past 20 trading days, the MACD line stayed **below the Signal Line ~65%** of the time.  
  - This means sellers have dominated, showing **consistent downward pressure**.  

- 🟠 **MACD Trending Downward:**  
  - The MACD line has dropped compared to 20 days ago.  
  - This suggests momentum is **fading**, with sellers slowly gaining control.  

- ⬇️ **Negative Histogram:**  
  - On average, the histogram stayed **below zero** during this period.  
  - This highlights that bearish momentum has been **heavier and persistent**.  

- ⚠️ **MACD Below Zero:**  
  - The MACD line is currently **below zero**, confirming that the stock is trading with **bearish momentum overall**.  
  - This often happens during **sustained downtrends**.  


### 🧠 What This Means for a Non-Stock Person  

- The stock is showing **strong selling pressure**.  
- Short-term and medium-term momentum are bearish.  
- It’s **not an ideal time to buy** unless there is a reversal signal (MACD crossing back above Signal).  
- Traders should be cautious, as momentum suggests the stock may continue downward.  

👉 **In simple words:**  
The MACD shows sellers are in control. Momentum is negative, and the stock is under pressure. Best to **avoid buying** right now unless a clear bullish reversal appears.  


## 4. Prophet

<p align="center">
<img width="1354" height="592" alt="image" src="https://github.com/user-attachments/assets/104f5f04-780b-4ba0-aae1-f9f9ee93511f"
</p>
  
## Prophet Forecast with Volume Insights  

The chart above shows **stock price forecasts** (blue line with confidence bands) along with **actual prices (black dots)** and **trading volume (red line)**.  

### 🔎 Key Observations  

1. **Trend Movements**  
   - From **Nov 2023 to mid-2024**, the stock moved **up strongly** (clear uptrend).  
   - From **mid-2024 to Jan 2025**, the stock entered a **downtrend** with falling prices.  
   - Currently, prices are showing **sideways/accumulation behavior** around the 650–700 range.  

2. **Volume Activity**  
   - High spikes in **volume** indicate periods of **strong buying or selling interest**.  
   - Recently, volume has been **rising**, but price has **not broken upward** significantly.  
   - This suggests **accumulation** (big players buying quietly). If price breaks resistance with strong volume, it may **start a new uptrend**.  

3. **Forecast & Confidence Intervals**  
   - Prophet projects prices for the **next 30 days** (blue forecast line).  
   - The **upper band** indicates potential bullish anomaly (if price breaks above → strong upside momentum).  
   - The **lower band** warns of bearish anomaly (if price breaks below → strong selling pressure).  

4. **Anomaly Detection**  
   - If actual price moves **outside the forecast bounds**, it signals an **anomaly**.  
   - 📈 *Above Upper Band* → Could mean **unexpected bullish breakout**.  
   - 📉 *Below Lower Band* → Could mean **unexpected bearish breakdown**.  

5. **Combined with Other Indicators**  
   - If **RSI > 50** and trending up → supports bullish case.  
   - If **SMA20 > SMA50** → confirms strength in short-term trend.  
   - If Prophet forecast is also upward → stronger confirmation of a rally.  
   - Conversely, weak RSI (<40) + SMA50 below SMA200 confirms continued bearish outlook.  

### 🧠 What This Means for a Non-Stock Person  

- The stock **went up, then down, and is now consolidating**.  
- Volume is rising, which often means **something big may happen soon**.  
- If the stock price **breaks above the forecast’s upper line with high volume**, it may start a **new rally (bullish)**.  
- If it **falls below the lower line**, it signals **further downside risk**.  
- By combining this with **RSI, SMA20/50 crossovers, and MACD**, traders can decide whether to **enter or wait**.  

👉 **In simple words:**  
This chart shows that the stock is at a **decision point**. High volume means pressure is building. If it breaks out above resistance, it could go up strongly. If it fails, it may go down more.  


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
