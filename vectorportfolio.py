import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# === Download Historical Data ===
tickers = ['AAPL', 'MSFT']
start_date = "2023-06-01"
end_date = "2024-06-01"

data = yf.download(tickers, start=start_date, end=end_date)
data.dropna(inplace=True)

# Check date count
N = len(data) - 1
t = np.arange(N+1)

# Portfolio weights
w = np.array([0.6, 0.4])

# Extract asset prices
S = data.values

# Initialize gradients
gradients = np.zeros((N+1, 2))
gradients[0] = [2 * w[0] * S[0,0], w[1] / (2 * np.sqrt(S[0,1]))]

# Compute gradients
for i in range(N):
    gradients[i+1, 0] = 2 * w[0] * S[i+1,0]
    gradients[i+1, 1] = w[1] / (2 * np.sqrt(S[i+1,1]))

# Compute gradient changes (numerical derivative)
grad_change = np.diff(gradients, axis=0)

# === Generate Signals ===
signals = []
threshold = 1.0

for g in grad_change:
    if np.sum(g) > threshold:
        signals.append(1)
    elif np.sum(g) < -threshold:
        signals.append(-1)
    else:
        signals.append(0)

signals.append(signals[-1])

signals = np.array(signals)

# === Compute Market & Strategy Returns ===
portfolio_value = w[0] * S[:,0] + w[1] * S[:,1]

market_returns = np.diff(portfolio_value) / portfolio_value[:-1]
strategy_returns = signals[:-1] * market_returns

# Cumulative returns
cum_market_return = np.cumprod(1 + market_returns) - 1
cum_strategy_return = np.cumprod(1 + strategy_returns) - 1
excess_return = cum_strategy_return[-1] - cum_market_return[-1]

# === Plot Prices with Signals ===
plt.figure(figsize=(14, 6))
plt.plot(data.index, S[:,0], label=tickers[0], color='blue')
plt.plot(data.index, S[:,1], label=tickers[1], color='orange')

# Plot buy/sell signals on AAPL
for i in range(N):
    if signals[i] == 1:
        plt.scatter(data.index[i], S[i,0], color='green', marker='^', s=70)
    elif signals[i] == -1:
        plt.scatter(data.index[i], S[i,0], color='red', marker='v', s=70)

# === Strategy Performance Stats ===
print("===== Strategy Performance (Real Data) =====") 
print(f"Final Market Return: {cum_market_return[-1]*100:.2f}%")
print(f"Final Strategy Return: {cum_strategy_return[-1]*100:.2f}%")
print(f"Excess Return (Strategy - Market): {excess_return*100:.2f}%")
print(f"Total Trades: {np.count_nonzero(signals)}")
print(f"Long Trades: {np.count_nonzero(signals==1)} | Short Trades: {np.count_nonzero(signals==-1)}")
print(f"Hit Ratio (Profitable Trades): {np.mean(strategy_returns > 0)*100:.2f}%")

plt.title("Asset Prices with Buy/Sell Signals")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()