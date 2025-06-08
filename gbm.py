import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Download historical price data (AAPL as example)
ticker = 'SPY'
data = yf.download(ticker, start="2023-01-01", end="2025-01-01")
prices = data['Close'].values

# Remove NaNs or non-finite values
prices = prices[np.isfinite(prices)]

# Confirm enough data for log returns
if len(prices) < 2:
    raise ValueError("Not enough price data downloaded for simulation.")

# Compute log returns
returns = np.diff(np.log(prices))

# Confirm enough returns to compute realized variance
if len(returns) < 2:
    raise ValueError("Not enough returns data to compute realized variance.")

# Compute realized variance (squared log returns)
realized_var = returns**2

# Confirm enough variance data
if len(realized_var) < 2:
    raise ValueError("Not enough realized variance data to estimate model parameters.")

# Time step (daily)
dt = 1 / 252

# Parameter estimation from data

# mu: mean log return per dt
mu = np.mean(returns) / dt

# rho: correlation between returns and variance changes
var_changes = np.diff(realized_var)
if len(var_changes) < 1:
    raise ValueError("Not enough variance change data to estimate rho.")
rho = np.corrcoef(returns[1:], var_changes)[0, 1]

# kappa, theta, xi estimation via method of moments on realized variance
v = realized_var
m1 = np.mean(v)
m2 = np.var(v)
delta_v = np.mean(np.diff(v)) / dt

if m1 - np.mean(v[:-1]) == 0:
    raise ValueError("Variance data not varying enough for kappa estimation.")

kappa = -delta_v / (m1 - np.mean(v[:-1]))
theta = m1
xi = np.sqrt(m2 / dt)

# Initial values
S0 = float(prices[-1])
v0 = float(v[-1])

# Simulation parameters
T = 1.0  # 1 year
N = 252  # number of time steps
dt = T / N

# Preallocate arrays
S = np.zeros(N + 1)
v_sim = np.zeros(N + 1)
t = np.linspace(0, T, N + 1)

# Initial conditions
S[0] = S0
v_sim[0] = v0

# Generate correlated Brownian increments
np.random.seed(42)
Z1 = np.random.normal(0, 1, N)
Z2 = np.random.normal(0, 1, N)
W1 = Z1 * np.sqrt(dt)
W2 = (rho * Z1 + np.sqrt(1 - rho ** 2) * Z2) * np.sqrt(dt)

# Euler-Maruyama Simulation (Exact to SDE)
for i in range(N):
    v_sim[i + 1] = np.abs(v_sim[i] + kappa * (theta - v_sim[i]) * dt + xi * np.sqrt(v_sim[i]) * W2[i])
    S[i + 1] = S[i] + mu * S[i] * dt + np.sqrt(v_sim[i]) * S[i] * W1[i]

# Plot simulation results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, S, label='Simulated Asset Price $S_t$')
plt.plot(t, prices, label=f"Actual {ticker} Prices")
plt.title(f'Stochastic Volatility GBM Simulation for {ticker}')
plt.ylabel('Asset Price')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, v_sim, label='Simulated Variance $v_t$', color='orange')
plt.xlabel('Time (Years)')
plt.ylabel('Variance')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Display estimated parameters
print(f"\nEstimated parameters from {ticker} historical data:")
print(f"mu    = {mu:.5f}")
print(f"kappa = {kappa:.5f}")
print(f"theta = {theta:.5f}")
print(f"xi    = {xi:.5f}")
print(f"rho   = {rho:.5f}")
