# Risk-Neutral Distribution Analyser

A Python tool for extracting market-implied probability distributions from options prices using the Breeden-Litzenberger method. Basically, I'm reverse-engineering what the market thinks will happen by analysing real option pricing data.

## 🎯 Why I'm Building This

I'm working on this project to get better at quantitative finance and eventually build my own algorithmic trading strategies. By extracting risk-neutral distributions from options markets, I can:

- **See what the market really thinks**: Get insight into collective trader expectations
- **Spot potential opportunities**: Compare what the market expects vs my own analysis
- **Learn the fundamentals**: Build up the maths and programming skills needed for systematic trading
- **Get hands-on experience**: Work with real financial data instead of just theory

## 🔬 What It Actually Does

The tool grabs live options data and applies the **Breeden-Litzenberger formula**:

```
Risk-Neutral Density = e^(rt) × ∂²C/∂K²
```

Where:
- `C` = Call option price
- `K` = Strike price  
- `r` = Risk-free rate
- `t` = Time to expiry
- `∂²C/∂K²` = Second derivative (calculated using finite differences)

This mathematical relationship lets us extract the probability distribution that's baked into current option prices.

## 📊 What It Can Do Right Now

- **Fetch real-time data** using the yfinance API
- **Calculate derivatives numerically** using finite difference methods
- **Filter dodgy data** to remove rubbish option prices
- **Create visualisations** comparing market-implied vs theoretical distributions
- **Crunch statistics** like skewness and kurtosis
- **Work with different assets** (SPY, QQQ, AAPL, etc.)

## 🚀 Example Output

![SPY Risk-Neutral Distribution](example_output.png)

*What the market thinks SPY will do - showing the expected price range and how likely each outcome is*

## 💻 Getting Started

```bash
# Grab the code
git clone https://github.com/yourusername/risk-neutral-distribution-analyser.git
cd risk-neutral-distribution-analyser

# Install what you need
pip install yfinance pandas numpy matplotlib scipy

# Run it
python main.py
```

## 🛠️ What I'm Planning to Add

### Next Month or Two
- [ ] **Volatility Smile Analysis**: Plot how implied vol changes across strikes
- [ ] **Multiple Expiry Comparison**: See how expectations change over time  
- [ ] **Historical Backtesting**: Check how good the market predictions actually were
- [ ] **Options Strategy Analyser**: Work out P&L for spreads and complex positions
- [ ] **Live Dashboard**: Real-time updating charts using Streamlit

### In a Few Months
- [ ] **Machine Learning Bits**: Try to predict distribution shapes from historical data
- [ ] **Cross-Asset Analysis**: Compare what's happening across different stocks/indices
- [ ] **Regime Detection**: Spot when markets are getting stressed
- [ ] **Monte Carlo Simulation**: Generate realistic price paths
- [ ] **Greeks Calculator**: Work out option sensitivities properly

### Longer Term (If I Don't Get Distracted!)
- [ ] **Signal Generation**: Actually find mispriced options
- [ ] **Portfolio Optimisation**: Use this stuff for proper risk management
- [ ] **Alternative Data**: Throw in sentiment, news, macro data
- [ ] **Algorithmic Trading Framework**: Build something that can actually trade
- [ ] **Write It Up**: Document everything properly (maybe even publish something)

## 📈 What I'm Learning

This project is helping me get better at:

- **Quantitative Finance**: Options theory, risk-neutral valuation, how markets actually work
- **Python Programming**: Proper OOP, numerical computing, handling messy data
- **Mathematical Methods**: Finite differences, numerical derivatives, statistics
- **Real Financial Data**: Dealing with missing data, outliers, and market quirks
- **Data Visualisation**: Making charts that actually tell a story

## 🎯 Why This Matters for My Career

I'm building this because it shows:

- **I'm serious about quant finance**: Not just reading about it, actually implementing it
- **I can code properly**: Clean, well-structured Python that does something useful
- **I understand the maths**: Not just using libraries blindly
- **I work with real data**: Not just toy examples from textbooks
- **I'm building towards something bigger**: This is a stepping stone to algorithmic trading

Plus it gives me something concrete to talk about in interviews rather than just "I'm interested in finance".

## 🔍 Technical Bits

**The Maths**: Breeden-Litzenberger theorem, finite difference methods, probability theory

**Main Libraries**: 
- `yfinance` - Getting market data
- `numpy` - Number crunching  
- `pandas` - Data wrangling
- `matplotlib` - Making pretty charts
- `scipy` - Scientific computing

**How It's Built**: Object-oriented design with separate classes for fetching data, doing calculations, and making visualisations.

## 📚 Useful Reading

- Breeden, D.T. and Litzenberger, R.H. (1978). "Prices of State-Contingent Claims Implicit in Option Prices"
- Jackwerth, J.C. (2004). "Option-Implied Risk-Neutral Distributions and Risk Aversion"
- Hull, J. (2018). "Options, Futures, and Other Derivatives"

## 🤝 Contributing

This is mainly a learning project for me, but if you spot bugs or have suggestions, feel free to open an issue or send a pull request!

## ⚠️ Important Note

This is just for learning and experimenting. Don't use it to make actual trading decisions without doing your own research first!