import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats

# === Load historical data ===
ticker = "JNJ"
start_date = "2023-01-01"
end_date = "2025-01-01"

data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

# Flatten MultiIndex columns if necessary
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [' '.join(col).strip() for col in data.columns.values]

print(data.head())
print(data.columns)

# Check what columns are present now
if 'Close' in data.columns:
    prices = data['Close'].dropna().values
elif f'Close {ticker}' in data.columns:
    prices = data[f'Close {ticker}'].dropna().values
else:
    raise ValueError("No valid price columns found in downloaded data.")

prices = prices.flatten()
dates = data.index[:len(prices)]

# === Improved Model Parameters ===
N = 5  # Number of agents
gamma = 2.0  # Reduced risk aversion for more reasonable positions
rho = 0.02  # Reduced mean reversion for longer-term positions
sigma_N = 0.005  # Much smaller noise volatility

# Calculate market statistics
log_returns = np.diff(np.log(prices))
sigma_D = np.std(log_returns)  # Daily volatility
mu = np.mean(log_returns)  # Mean daily return

print(f"Market statistics:")
print(f"Daily volatility: {sigma_D:.4f}")
print(f"Mean daily return: {mu:.6f}")
print(f"Annualized volatility: {sigma_D * np.sqrt(252):.2%}")
print(f"Annualized return: {mu * 252:.2%}")

# === Enhanced Model Price Calculation ===
steps = len(prices)
dt = 1.0  # Daily time step
np.random.seed(42)

# Initialize arrays
pt = np.zeros(steps)  # Model price
fundamental_value = np.zeros(steps)  # Fundamental value estimate
sentiment = np.zeros(steps)  # Market sentiment
X = np.zeros((N, steps))  # Individual agent holdings

# Calculate fundamental value using multiple indicators
def calculate_fundamental_value(prices, window=50):
    """Calculate fundamental value using technical indicators"""
    fv = np.zeros(len(prices))
    
    for i in range(len(prices)):
        start_idx = max(0, i - window + 1)
        end_idx = i + 1
        
        if i < window:
            # For early periods, use simple moving average
            fv[i] = np.mean(prices[start_idx:end_idx])
        else:
            # Use exponential moving average with multiple timeframes
            short_ema = prices[i-9:i+1]
            long_ema = prices[i-29:i+1]
            
            # Calculate EMAs
            alpha_short = 2.0 / (10 + 1)
            alpha_long = 2.0 / (30 + 1)
            
            ema_short = np.mean(short_ema)  # Simplified EMA
            ema_long = np.mean(long_ema)
            
            # Combine with price momentum and volume-weighted consideration
            momentum = (prices[i] - prices[i-20]) / prices[i-20] if i >= 20 else 0
            
            # Fundamental value as weighted average of indicators
            fv[i] = 0.4 * ema_long + 0.3 * ema_short + 0.3 * prices[i] * (1 - 0.1 * momentum)
    
    return fv

# Calculate fundamental values
fundamental_value = calculate_fundamental_value(prices)

# Generate market sentiment (Ornstein-Uhlenbeck process)
psi = 0.05  # Sentiment mean reversion
sentiment[0] = 0
dB_sentiment = np.random.normal(0, np.sqrt(dt), steps)

for t in range(1, steps):
    # Sentiment influenced by recent returns
    recent_return = (prices[t] - prices[t-1]) / prices[t-1]
    sentiment[t] = sentiment[t-1] * (1 - psi * dt) + sigma_N * dB_sentiment[t] + 0.5 * recent_return

# === Multi-Agent Dynamics ===
# Initialize agent holdings
for n in range(N):
    X[n, 0] = np.random.normal(0, 0.1)  # Small random initial positions

# Calculate model price with improved dynamics
pt[0] = prices[0]  # Initialize with market price

for t in range(1, steps):
    # Update agent holdings based on fundamental value and sentiment
    total_demand = 0
    
    for n in range(N):
        # Each agent's view of fair value
        agent_bias = 0.02 * (n - N/2) / N  # Agents have different biases
        perceived_value = fundamental_value[t] * (1 + agent_bias + 0.1 * sentiment[t])
        
        # Optimal position based on perceived mispricing
        mispricing = (perceived_value - prices[t]) / prices[t]
        
        # Position adjustment with mean reversion
        target_position = mispricing / (gamma * sigma_D**2)
        target_position = np.clip(target_position, -2.0, 2.0)  # Limit position size
        
        # Gradual adjustment toward target
        adjustment_speed = 0.1
        X[n, t] = X[n, t-1] + adjustment_speed * (target_position - X[n, t-1])
        
        total_demand += X[n, t]
    
    # Model price includes impact of agent demand
    demand_impact = total_demand * sigma_D * 0.1  # Scaled demand impact
    noise_impact = sentiment[t] * sigma_D * 0.5
    
    # Model price evolution
    pt[t] = fundamental_value[t] + demand_impact + noise_impact

# === Improved Signal Generation ===
# Calculate z-score of price difference
price_diff = pt - prices
lookback_window = 60  # 3-month lookback for statistics

signals = np.zeros(steps)
z_scores = np.zeros(steps)

for t in range(lookback_window, steps):
    # Calculate rolling statistics
    recent_diffs = price_diff[t-lookback_window:t]
    mean_diff = np.mean(recent_diffs)
    std_diff = np.std(recent_diffs)
    
    if std_diff > 0:
        z_scores[t] = (price_diff[t] - mean_diff) / std_diff
        
        # Generate signals based on z-score thresholds
        if z_scores[t] < -1.5:  # Model significantly below market (buy)
            signals[t] = 1
        elif z_scores[t] > 1.5:  # Model significantly above market (sell)
            signals[t] = -1
        elif abs(z_scores[t]) < 0.5:  # Prices converged, exit position
            signals[t] = 0
        else:
            signals[t] = 0  # Hold current position

# === Enhanced Position Management ===
positions = np.zeros(steps)
position_sizes = np.zeros(steps)

# Risk management parameters
max_position = 1.0
position_decay = 0.95  # Daily position decay when no signal

for t in range(1, steps):
    if signals[t] == 1:  # Buy signal
        # Position size based on signal strength
        signal_strength = min(abs(z_scores[t]), 3.0) / 3.0
        target_position = signal_strength * max_position
        positions[t] = target_position
    elif signals[t] == -1:  # Sell signal
        signal_strength = min(abs(z_scores[t]), 3.0) / 3.0
        target_position = -signal_strength * max_position
        positions[t] = target_position
    else:
        # Decay position when no signal
        positions[t] = positions[t-1] * position_decay
        
    # Close very small positions
    if abs(positions[t]) < 0.05:
        positions[t] = 0
    
    position_sizes[t] = abs(positions[t])

# === Performance Calculation ===
# Market returns
market_returns = np.zeros(steps)
market_returns[1:] = np.diff(prices) / prices[:-1]

# Strategy returns with transaction costs
transaction_cost = 0.001  # 0.1% per trade
strategy_returns = np.zeros(steps)
transaction_costs = np.zeros(steps)

for t in range(1, steps):
    # Return from position
    strategy_returns[t] = positions[t-1] * market_returns[t]
    
    # Transaction costs
    position_change = abs(positions[t] - positions[t-1])
    if position_change > 0.01:  # Significant position change
        transaction_costs[t] = position_change * transaction_cost
        strategy_returns[t] -= transaction_costs[t]

# Cumulative returns
cum_market = np.cumprod(1 + market_returns) - 1
cum_strategy = np.cumprod(1 + strategy_returns) - 1

# === Performance Analysis ===
total_trades = np.sum(np.abs(np.diff(positions)) > 0.01)
total_transaction_costs = np.sum(transaction_costs)
long_trades = np.sum(np.diff(positions) > 0.01)
short_trades = np.sum(np.diff(positions) < -0.01)

final_market_return = cum_market[-1]
final_strategy_return = cum_strategy[-1]

print(f"\n=== Performance Summary ===")
print(f"Total trades: {total_trades}")
print(f"Long trades: {long_trades}, Short trades: {short_trades}")
print(f"Total transaction costs: {total_transaction_costs:.2%}")
print(f"Final market return: {final_market_return:.2%}")
print(f"Final strategy return: {final_strategy_return:.2%}")
print(f"Excess return: {(final_strategy_return - final_market_return):.2%}")

# Risk metrics
market_vol = np.std(market_returns[1:]) * np.sqrt(252)
strategy_vol = np.std(strategy_returns[1:]) * np.sqrt(252)
market_sharpe = np.mean(market_returns[1:]) / np.std(market_returns[1:]) * np.sqrt(252) if np.std(market_returns[1:]) > 0 else 0
strategy_sharpe = np.mean(strategy_returns[1:]) / np.std(strategy_returns[1:]) * np.sqrt(252) if np.std(strategy_returns[1:]) > 0 else 0

print(f"\n=== Risk Metrics ===")
print(f"Market volatility: {market_vol:.2%}")
print(f"Strategy volatility: {strategy_vol:.2%}")
print(f"Market Sharpe ratio: {market_sharpe:.2f}")
print(f"Strategy Sharpe ratio: {strategy_sharpe:.2f}")

# Maximum drawdown
def calculate_max_drawdown(returns):
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    return np.min(drawdown)

market_dd = calculate_max_drawdown(market_returns[1:])
strategy_dd = calculate_max_drawdown(strategy_returns[1:])
print(f"Market max drawdown: {market_dd:.2%}")
print(f"Strategy max drawdown: {strategy_dd:.2%}")

# === Plotting ===
plt.figure(figsize=(16, 12))

plt.subplot(5,1,1)
plt.plot(dates, prices, label='Market Price', alpha=0.8, linewidth=1.5)
plt.plot(dates, pt, label='Model Price', alpha=0.8, linewidth=1.5)
plt.plot(dates, fundamental_value, label='Fundamental Value', alpha=0.6, linewidth=1)
plt.legend()
plt.title('Prices and Model Estimates')
plt.grid(True, alpha=0.3)

plt.subplot(5,1,2)
plt.plot(dates, z_scores, label='Z-Score (Model-Market)', color='purple', linewidth=1)
plt.axhline(y=1.5, color='r', linestyle='--', alpha=0.7, label='Buy/Sell Thresholds')
plt.axhline(y=-1.5, color='r', linestyle='--', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.legend()
plt.title('Standardized Price Difference (Z-Score)')
plt.grid(True, alpha=0.3)

plt.subplot(5,1,3)
plt.plot(dates, sentiment, label='Market Sentiment', color='orange', alpha=0.8)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.legend()
plt.title('Market Sentiment')
plt.grid(True, alpha=0.3)

plt.subplot(5,1,4)
plt.plot(dates, positions, label='Position Size', drawstyle='steps-post', linewidth=1.5)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.legend()
plt.title('Trading Positions')
plt.ylim(-1.1, 1.1)
plt.grid(True, alpha=0.3)

plt.subplot(5,1,5)
plt.plot(dates, cum_market * 100, label='Market Return (%)', linewidth=1.5)
plt.plot(dates, cum_strategy * 100, label='Strategy Return (%)', linewidth=1.5)
plt.legend()
plt.title('Cumulative Returns')
plt.ylabel('Return (%)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Signal Analysis ===
signal_stats = pd.Series(signals).value_counts().sort_index()
print(f"\n=== Signal Distribution ===")
for signal, count in signal_stats.items():
    signal_name = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}.get(signal, f'Signal_{signal}')
    print(f"{signal_name}: {count} ({count/len(signals):.1%})")

# Show statistics for signal generation
non_zero_signals = signals[signals != 0]
if len(non_zero_signals) > 0:
    print(f"\nAverage signal strength: {np.mean(np.abs(z_scores[signals != 0])):.2f}")
    print(f"Signal frequency: {len(non_zero_signals)/len(signals):.1%}")
else:
    print("\nNo trading signals generated")

# Performance attribution
avg_position_size = np.mean(position_sizes[position_sizes > 0])
print(f"\nAverage position size: {avg_position_size:.2f}")
print(f"Position utilization: {np.mean(position_sizes):.2f}")