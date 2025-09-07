"""
Analyse option data to extract risk-neutral distributions using the Breeden-Litzernberger method
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# OptionAnalyser Class - handles data fetching and processing
class OptionAnalyser:

    def __init__(self, symbol):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
    
    def get_available_expirations(self):
        expirations = self.ticker.options
        return list(expirations)

    def get_option_chain(self, expiry_date):
        
        # Get Option Chain
        option_chain = self.ticker.option_chain(expiry_date)
        calls = option_chain.calls
        puts = option_chain.puts

        # Gets the latest historical close data (open, high, low, close, volume) as a Pd dataFrame
        close_price = self.ticker.history(period='1d')['Close'].iloc[-1]
        return {
            'calls': calls,  # Added this - you were missing the actual calls data
            'puts': puts,    # Added this - you were missing the actual puts data
            'current_price': close_price,
            'expiry': expiry_date
        }

# RiskNeutralDistribution Class - Implements Breeden-Litzenberger method for risk-neutral distribution
class RiskNeutralDistribution:

    def __init__(self, strikes, call_prices, risk_free_rate=0.05, time_to_expiry=None):
        self.strikes = np.array(strikes)
        self.call_prices = np.array(call_prices)
        self.risk_free_rate = risk_free_rate
        self.time_to_expiry = time_to_expiry

        # Sort by strike price, argsort gives you the indices that would sort the array
        # Sorts option strikes into ascending order, reorders call prices to match strike
        sort_idx = np.argsort(self.strikes)
        self.strikes = self.strikes[sort_idx]
        self.call_prices = self.call_prices[sort_idx]
    
    def compute_risk_neutral_density(self, smooth=True):  # Fixed typo: was "netural"
        # Compute risk-neutral density using Breeden-Litzenberger formula
        # BL Formila relates the second derivative of a call option's price to the underlying asset's
        # risk neutral probability density function, allowing for the extraction of the market's
        # implied probability distirbution from observable option prices
        
        # Interpolate to get uniform strike spacing if needed
        if smooth:
            strikes_uniform, call_prices_uniform = self._interpolate_prices()
        else:
            strikes_uniform = self.strikes
            call_prices_uniform = self.call_prices
        
        # Calculate second derivative using finite differences
        second_derivative = self._finiteDifference(strikes_uniform, call_prices_uniform)

        # Appply Breeden-Litzenberger Formula
        discount_factor = np.exp(self.risk_free_rate * self.time_to_expiry)
        risk_neutral_density = discount_factor * second_derivative

        # Remove negative densities (can occure due to noise)
        risk_neutral_density = np.maximum(risk_neutral_density, 0)

        return strikes_uniform[1:-1], risk_neutral_density # Remove edge points

    
    def _interpolate_prices(self, num_points=None):
        # Interpolate call prices to uniform strike grid
        if num_points is None:
            num_points = len(self.strikes)
        
        # Create uniform strike grid
        strike_min = self.strikes.min()
        strike_max = self.strikes.max()
        strikes_uniform = np.linspace(strike_min, strike_max, num_points)

        # Interpolate call prices
        interp_func = interp1d(self.strikes, self.call_prices, kind='cubic', bounds_error=False, fill_value='extrapolate')
        call_prices_uniform = interp_func(strikes_uniform)

        return strikes_uniform, call_prices_uniform

    
    def _finiteDifference(self, strikes, prices):
        # Calculate second derivative using central difference method

        n = len(prices)
        second_deriv = np.zeros(n-2)

        for i in range(1, n-1):
            # Calculate step sizes (handles uneven spacing)
            h1 = strikes[i] - strikes[i-1] # backward step
            h2 = strikes[i+1] - strikes[i] # forward step

            # Central difference formula for uneven spacing
            second_deriv[i-1] = (2 * (prices[i-1]/(h1*(h1+h2)) - prices[i]/(h1*h2) + prices[i+1]/(h2*(h1+h2))) )
    
        return second_deriv
    
    def calculate_stats(self, strikes, density):
        # Calcualtes key statistics of the risk-neutral distribution

        # Normalise density
        dx = strikes[1] - strikes[0] if len(strikes) > 1 else 1
        density_norm = density / (np.sum(density) * dx)

        # Calcaulte moments
        mean = np.sum(strikes * density_norm * dx)
        variance = np.sum((strikes - mean)**2 * density_norm * dx)
        std = np.sqrt(variance)

        # Skewness and Kurtosis
        skewness = np.sum((strikes - mean)**3 * density_norm * dx) / (std**3)
        kurtosis = np.sum((strikes - mean)**4 * density_norm * dx) / (std**4) - 3

        return {
            'mean': mean,
            'std': std,
            'variance': variance,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

def plot_results(strikes, density, current_price, symbol, expiry, stats):
    # Plot the risk neutral distribution
    plt.figure(figsize=(12,6))
    
    # Plot risk-neutral density
    plt.plot(strikes, density, 'b-', linewidth=2, label='Risk-Neutral Density')
    
    # Add log-normal distribution for comparison if stats available
    if stats:
        # Calculate log-normal distribution using current price as forward price
        # Use a simple volatility estimate from the RND standard deviation
        S0 = current_price
        sigma = stats['std'] / S0  # Convert to relative volatility
        
        # Create log-normal density: f(S) = 1/(S*sigma*sqrt(2*pi)) * exp(-(ln(S/S0))^2 / (2*sigma^2))
        log_normal_density = (1 / (strikes * sigma * np.sqrt(2 * np.pi))) * \
                           np.exp(-(np.log(strikes / S0))**2 / (2 * sigma**2))
        
        # Scale the log-normal to match the peak of the RND for better comparison
        scale_factor = np.max(density) / np.max(log_normal_density)
        log_normal_density = log_normal_density * scale_factor
        
        plt.plot(strikes, log_normal_density, 'g:', linewidth=2, alpha=0.8, 
                label='Log-Normal Approx')
    
    # Add current price line
    plt.axvline(current_price, color='r', linestyle='--', alpha=0.7, 
                label=f'Current Price: ${current_price:.2f}')
    
    plt.xlabel('Strike Price ($)')
    plt.ylabel('Probability Density')
    plt.title(f'{symbol} Risk-Neutral Distribution - Expiry: {expiry}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    if stats:
        print(f"\nRisk-Neutral Distribution Statistics:")
        print(f"Mean: ${stats['mean']:.2f}")
        print(f"Standard Deviation: ${stats['std']:.2f}")
        print(f"Skewness: {stats['skewness']:.3f}")
        print(f"Kurtosis: {stats['kurtosis']:.3f}")

def calculate_time_to_expiry(expiry_date):
    """Calculate time to expiry in years"""
    expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
    now = datetime.now()
    days_to_expiry = (expiry - now).days
    return max(days_to_expiry / 365.0, 1/365)  # Minimum 1 day

# Simple main function
def main():
    # Create analyzer for SPY
    analyzer = OptionAnalyser("SPY")
    
    # Get available expirations
    expirations = analyzer.get_available_expirations()
    print(f"Available expirations: {expirations[:3]}")  # Show first 3
    
    # Use first expiration
    expiry = expirations[0]
    print(f"Using expiry: {expiry}")
    
    # Get option data
    option_data = analyzer.get_option_chain(expiry)
    calls = option_data['calls']
    current_price = option_data['current_price']
    
    print(f"Current SPY price: ${current_price:.2f}")
    print(f"Number of call options: {len(calls)}")
    
    # Filter calls - use only options with some volume
    calls_filtered = calls[calls['volume'] > 0]
    print(f"Calls with volume > 0: {len(calls_filtered)}")
    
    # Use mid price for analysis
    calls_filtered['mid_price'] = (calls_filtered['bid'] + calls_filtered['ask']) / 2
    
    # Calculate time to expiry
    time_to_expiry = calculate_time_to_expiry(expiry)
    print(f"Time to expiry: {time_to_expiry:.3f} years")
    
    # Create RND calculator
    rnd_calc = RiskNeutralDistribution(
        strikes=calls_filtered['strike'].values,
        call_prices=calls_filtered['mid_price'].values,
        risk_free_rate=0.05,
        time_to_expiry=time_to_expiry
    )
    
    # Compute risk-neutral distribution
    strikes, density = rnd_calc.compute_risk_neutral_density()
    
    # Calculate statistics
    stats = rnd_calc.calculate_stats(strikes, density)
    
    # Plot results
    plot_results(strikes, density, current_price, "SPY", expiry, stats)

if __name__ == "__main__":
    main()