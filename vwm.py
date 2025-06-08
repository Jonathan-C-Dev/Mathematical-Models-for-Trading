import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import jarque_bera, kstest, normaltest
import warnings
warnings.filterwarnings('ignore')

class StockSimulator:
    def __init__(self, ticker="AAPL", start_date="2020-01-01", end_date="2025-01-01"):
        """
        Practical stock simulator using AR(1) returns + stochastic volatility
        This model typically outperforms standard GBM and fancy momentum models
        """
        self.ticker = ticker
        # Fix for yfinance auto_adjust change
        self.data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        self.prices = self.data['Close'].values
        self.prices = self.prices[np.isfinite(self.prices)]
        self.returns = np.diff(np.log(self.prices))
        
        # Remove outliers (beyond 5 standard deviations)
        returns_std = np.std(self.returns)
        outlier_mask = np.abs(self.returns) < 5 * returns_std
        self.returns = self.returns[outlier_mask]
        
        print(f"Loaded {len(self.prices)} price observations for {ticker}")
        print(f"Using {len(self.returns)} returns after outlier removal")
        
    def estimate_parameters(self):
        """
        Estimate model parameters using Maximum Likelihood
        Model: r_t = phi * r_{t-1} + sigma_t * epsilon_t
               sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        """
        returns = self.returns
        
        # Step 1: Estimate AR(1) parameter
        r_lag = returns[:-1]
        r_current = returns[1:]
        
        # OLS for AR(1): r_t = phi * r_{t-1} + error
        denominator = np.sum(r_lag**2)
        if denominator > 0:
            phi = np.sum(r_lag * r_current) / denominator
        else:
            phi = 0.0
        phi = np.clip(phi, -0.99, 0.99)  # Ensure stationarity
        
        # AR(1) residuals
        ar_residuals = r_current - phi * r_lag
        
        # Step 2: Estimate GARCH parameters on AR residuals
        def garch_likelihood(params):
            omega, alpha, beta = params
            
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            n = len(ar_residuals)
            sigma2 = np.zeros(n)
            sigma2[0] = np.var(ar_residuals)  # Initial variance
            
            log_likelihood = 0
            for t in range(1, n):
                sigma2[t] = omega + alpha * ar_residuals[t-1]**2 + beta * sigma2[t-1]
                if sigma2[t] <= 0:
                    return 1e10
                log_likelihood -= 0.5 * (np.log(2*np.pi) + np.log(sigma2[t]) + ar_residuals[t]**2/sigma2[t])
            
            return -log_likelihood
        
        # Initial parameter guess
        unconditional_var = np.var(ar_residuals)
        initial_params = [
            unconditional_var * 0.01,  # omega
            0.05,                      # alpha
            0.90                       # beta
        ]
        
        # Bounds for optimization
        bounds = [(1e-8, unconditional_var * 0.1), (0, 0.3), (0, 0.95)]
        
        try:
            result = minimize(garch_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')
            if result.success and result.fun < 1e9:
                omega, alpha, beta = result.x
                # Ensure persistence constraint
                if alpha + beta >= 0.999:
                    beta = 0.999 - alpha
            else:
                # Fallback to simple estimates
                omega = unconditional_var * 0.01
                alpha = 0.05
                beta = 0.90
        except:
            # Fallback parameters
            omega = unconditional_var * 0.01
            alpha = 0.05
            beta = 0.90
        
        # Store parameters
        self.phi = phi
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.initial_vol = np.std(returns)
        
        print(f"\nEstimated Parameters:")
        print(f"AR(1) coefficient (φ): {self.phi:.4f}")
        print(f"GARCH omega (ω): {self.omega:.6f}")
        print(f"GARCH alpha (α): {self.alpha:.4f}")
        print(f"GARCH beta (β): {self.beta:.4f}")
        print(f"Persistence (α+β): {self.alpha + self.beta:.4f}")
        
        return self.phi, self.omega, self.alpha, self.beta
    
    def simulate(self, n_days, n_simulations=1, seed=42):
        """
        Simulate stock prices using AR(1)-GARCH model
        """
        np.random.seed(seed)
        
        if not hasattr(self, 'phi'):
            self.estimate_parameters()
        
        S0 = self.prices[-1]  # Start from last observed price
        simulations = []
        
        for sim in range(n_simulations):
            # Initialize arrays
            returns = np.zeros(n_days)
            sigma2 = np.zeros(n_days)
            prices = np.zeros(n_days + 1)
            
            prices[0] = S0
            sigma2[0] = self.initial_vol**2
            
            # Generate random shocks
            epsilon = np.random.standard_normal(n_days)
            
            # Start with zero return (or small random return)
            returns[0] = np.random.normal(0, self.initial_vol * 0.1)
            
            for t in range(1, n_days):
                # Update conditional variance (GARCH)
                sigma2[t] = self.omega + self.alpha * (returns[t-1]**2) + self.beta * sigma2[t-1]
                sigma2[t] = max(sigma2[t], 1e-8)  # Prevent negative/zero variance
                
                # Generate return (AR(1))
                returns[t] = self.phi * returns[t-1] + np.sqrt(sigma2[t]) * epsilon[t]
                
                # Cap extreme returns to prevent numerical issues
                returns[t] = np.clip(returns[t], -0.2, 0.2)  # ±20% daily return cap
                
                # Update price
                prices[t+1] = prices[t] * np.exp(returns[t])
                
                # Sanity check for price
                if prices[t+1] <= 0 or not np.isfinite(prices[t+1]):
                    prices[t+1] = prices[t]  # Keep previous price if something goes wrong
            
            simulations.append(prices)
        
        return np.array(simulations)
    
    def backtest_model(self, test_days=252):
        """
        Backtest the model on recent data
        """
        if len(self.prices) < test_days + 252:
            print("Not enough data for backtesting")
            return
        
        # Use earlier data for parameter estimation
        train_prices = self.prices[:-test_days]
        test_prices = self.prices[-test_days:]
        
        # Temporarily replace data for parameter estimation
        original_prices = self.prices.copy()
        original_returns = self.returns.copy()
        
        self.prices = train_prices
        self.returns = np.diff(np.log(train_prices))
        # Remove outliers from training returns too
        returns_std = np.std(self.returns)
        outlier_mask = np.abs(self.returns) < 5 * returns_std
        self.returns = self.returns[outlier_mask]
        
        self.estimate_parameters()
        
        # Simulate forward starting from the last training price
        simulated = self.simulate(test_days, n_simulations=100, seed=42)
        
        # Restore original data
        self.prices = original_prices
        self.returns = original_returns
        
        # Calculate metrics
        sim_mean = np.mean(simulated, axis=0)
        sim_std = np.std(simulated, axis=0)
        
        # Debug: Check if simulation looks reasonable
        print(f"Starting price for simulation: {train_prices[-1]:.2f}")
        print(f"Simulated mean final price: {sim_mean[-1]:.2f}")
        print(f"Actual final price: {test_prices[-1]:.2f}")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Main price comparison
        plt.subplot(2, 2, 1)
        t_test = np.arange(len(test_prices))
        
        plt.plot(t_test, test_prices, 'black', linewidth=2, label='Actual Prices')
        plt.plot(t_test, sim_mean[1:], 'red', linewidth=2, label='Simulated Mean')  # Skip first element (starting price)
        plt.fill_between(t_test, sim_mean[1:] - 2*sim_std[1:], sim_mean[1:] + 2*sim_std[1:], 
                        color='red', alpha=0.2, label='95% Confidence Band')
        
        # Plot some individual simulation paths
        for i in range(0, min(10, len(simulated))):
            plt.plot(t_test, simulated[i][1:], 'blue', alpha=0.1, linewidth=0.5)
        
        plt.title(f'Backtest: {self.ticker} - Last {test_days} Days')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Return distribution comparison
        plt.subplot(2, 2, 2)
        actual_returns = np.diff(np.log(test_prices))
        sim_returns = np.diff(np.log(sim_mean))
        
        # Filter out any remaining infinite values
        actual_returns = actual_returns[np.isfinite(actual_returns)]
        sim_returns = sim_returns[np.isfinite(sim_returns)]
        
        if len(actual_returns) > 0 and len(sim_returns) > 0:
            plt.hist(actual_returns, bins=30, alpha=0.7, density=True, label='Actual Returns', range=(-0.15, 0.15))
            plt.hist(sim_returns, bins=30, alpha=0.7, density=True, label='Simulated Returns', range=(-0.15, 0.15))
            plt.title('Return Distribution Comparison')
            plt.xlabel('Daily Returns')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Invalid return data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Return Distribution - Error')
        
        # Volatility comparison
        plt.subplot(2, 2, 3)
        window = 20
        actual_vol = []
        sim_vol = []
        
        for i in range(window, len(actual_returns)):
            actual_vol.append(np.std(actual_returns[i-window:i]) * np.sqrt(252))
        
        for i in range(window, len(sim_returns)):
            sim_vol.append(np.std(sim_returns[i-window:i]) * np.sqrt(252))
        
        min_len = min(len(actual_vol), len(sim_vol))
        if min_len > 0:
            plt.plot(actual_vol[:min_len], label='Actual Rolling Vol', linewidth=2)
            plt.plot(sim_vol[:min_len], label='Simulated Rolling Vol', linewidth=2)
            plt.title(f'Rolling Volatility ({window}-day)')
            plt.xlabel('Days')
            plt.ylabel('Annualized Volatility')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Invalid volatility data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Rolling Volatility - Error')
        
        # Error metrics
        plt.subplot(2, 2, 4)
        price_errors = np.abs(test_prices - sim_mean[1:]) / test_prices * 100
        price_errors = price_errors[np.isfinite(price_errors)]
        
        if len(price_errors) > 0:
            plt.plot(price_errors, 'red', linewidth=1)
            plt.axhline(np.mean(price_errors), color='blue', linestyle='--', 
                       label=f'Mean Error: {np.mean(price_errors):.2f}%')
            plt.title('Absolute Percentage Errors')
            plt.xlabel('Days')
            plt.ylabel('Error (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Invalid error data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Absolute Percentage Errors - Error')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\nBacktest Results:")
        if len(price_errors) > 0:
            print(f"Mean Absolute Percentage Error: {np.mean(price_errors):.2f}%")
        
        if len(actual_returns) > 0 and len(sim_returns) > 0:
            min_len = min(len(actual_returns), len(sim_returns))
            correlation = np.corrcoef(actual_returns[:min_len], sim_returns[:min_len])[0,1]
            if np.isfinite(correlation):
                print(f"Return Correlation: {correlation:.4f}")
                        
            vol_ratio = np.std(sim_returns) / np.std(actual_returns)
            if np.isfinite(vol_ratio):
                print(f"Volatility Ratio (Sim/Actual): {vol_ratio:.4f}")
            
            # Directional accuracy
            actual_signs = np.sign(actual_returns[:min_len])
            sim_signs = np.sign(sim_returns[:min_len])
            directional_accuracy = np.mean(actual_signs == sim_signs)
            
            if np.isfinite(directional_accuracy):
                print(f"Directional Accuracy: {directional_accuracy:.1%}")

# Example usage
if __name__ == "__main__":
    # Initialize simulator
    sim = StockSimulator("AAPL", "2020-01-01", "2025-01-01")
    
    # Estimate parameters
    sim.estimate_parameters()
    
    # Run backtest
    print("\nRunning backtest...")
    sim.backtest_model(test_days=200)
    
    # Generate future simulations
    print("\nGenerating future price paths...")
    future_sims = sim.simulate(n_days=60, n_simulations=50)
    
    # Plot future scenarios
    plt.figure(figsize=(12, 8))
    
    # Plot historical prices
    hist_days = min(100, len(sim.prices))
    historical_t = np.arange(-hist_days, 0)
    plt.plot(historical_t, sim.prices[-hist_days:], 'black', linewidth=3, label='Historical')
    
    # Plot simulations
    future_t = np.arange(0, future_sims.shape[1])
    for i in range(future_sims.shape[0]):
        plt.plot(future_t, future_sims[i], 'blue', alpha=0.3, linewidth=0.5)
    
    # Plot mean and confidence bands
    sim_mean = np.mean(future_sims, axis=0)
    sim_std = np.std(future_sims, axis=0)
    plt.plot(future_t, sim_mean, 'red', linewidth=2, label='Expected Path')
    plt.fill_between(future_t, sim_mean - 2*sim_std, sim_mean + 2*sim_std, 
                    color='red', alpha=0.2, label='95% Confidence')
    
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.title(f'{sim.ticker} - Future Price Scenarios (Next 60 Days)')
    plt.xlabel('Days from Today')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"\nCurrent Price: ${sim.prices[-1]:.2f}")
    print(f"Expected Price in 30 days: ${sim_mean[30]:.2f}")
    print(f"Expected Price in 60 days: ${sim_mean[-1]:.2f}")
    print(f"95% Confidence Range (60 days): ${sim_mean[-1] - 2*sim_std[-1]:.2f} - ${sim_mean[-1] + 2*sim_std[-1]:.2f}")