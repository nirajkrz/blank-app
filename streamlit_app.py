import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# Page configuration
st.set_page_config(
    page_title="Dual Moving Average Trading Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class LeveragedETFStrategy:
    """
    Leveraged ETF Trading Strategy using TQQQ/SQQQ w                # Chart 2: Moving Average Analysis
                fig2 = go.Figure()
                
                # Calculate MA distances for plotting
                ma_slow_dist = ((data['Close'] - data['SMA_250']) / data['SMA_250'] * 100).fillna(0)
                ma_fast_dist = ((data['Close'] - data['SMA_50']) / data['SMA_50'] * 100).fillna(0)
                
                fig2.add_trace(go.Scatter(x=data.index, y=ma_slow_dist,
                                         name='Distance from 250-day MA',
                                         line=dict(color='purple', width=2)))
                fig2.add_trace(go.Scatter(x=data.index, y=ma_fast_dist,
                                         name='Distance from 50-day MA',
                                         line=dict(color='orange', width=2)))0-day and 250-day MAs
    
    Long Signal (Buy TQQQ):
    - 50-day MA crosses above 250-day MA (Golden Cross)
    - Price is above both MAs for confirmation
    
    Short Signal (Switch to SQQQ):
    - 50-day MA crosses below 250-day MA (Death Cross)
    - Price is below both MAs for confirmation
    """
    
    def __init__(self, base_ticker='QQQ', fast_ma=50, slow_ma=250, 
                 profit_target=0.15, stop_loss=0.10, roc_period=20):
        """Initialize the strategy parameters"""
        self.base_ticker = base_ticker  # QQQ for tracking signals
        self.long_ticker = 'TQQQ'  # 3x leveraged long
        self.short_ticker = 'SQQQ'  # 3x leveraged short
        self.fast_ma = fast_ma  # 50-day MA
        self.slow_ma = slow_ma  # 250-day MA
        self.profit_target = profit_target  # 15% profit target (adjusted for leverage)
        self.stop_loss = stop_loss  # 10% stop loss (adjusted for leverage)
        self.roc_period = roc_period
        self.data = None
        self.trades = []
        
    def fetch_data(self, start_date=None, end_date=None):
        """Fetch historical price data for QQQ, TQQQ, and SQQQ"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Fetch QQQ data for signal generation
        base_data = yf.download(self.base_ticker, start=start_date, end=end_date, progress=False)
        
        # Fetch TQQQ and SQQQ data for actual trading
        tqqq_data = yf.download(self.long_ticker, start=start_date, end=end_date, progress=False)
        sqqq_data = yf.download(self.short_ticker, start=start_date, end=end_date, progress=False)
        
        if len(base_data) == 0:
            raise ValueError(f"No data found for {self.base_ticker}")
        
        # Combine the data
        self.data = base_data.copy()
        self.data['TQQQ_Close'] = tqqq_data['Close']
        self.data['TQQQ_Open'] = tqqq_data['Open']
        self.data['SQQQ_Close'] = sqqq_data['Close']
        self.data['SQQQ_Open'] = sqqq_data['Open']
        
        years_of_data = (self.data.index[-1] - self.data.index[0]).days / 365.25
        
        return self.data, years_of_data
    
    def calculate_indicators(self):
        """Calculate technical indicators"""
        df = self.data.copy()
        
        # Fast and Slow Moving Averages
        df['SMA_50'] = df['Close'].rolling(window=self.fast_ma).mean()
        df['SMA_250'] = df['Close'].rolling(window=self.slow_ma).mean()
        
        # Calculate distances safely
        close_prices = df['Close'].values
        sma_50_values = df['SMA_50'].values
        sma_250_values = df['SMA_250'].values
        
        # Calculate distances using numpy operations
        df.loc[:, 'MA_Fast_Dist'] = ((close_prices - sma_50_values) / sma_50_values) * 100
        df.loc[:, 'MA_Slow_Dist'] = ((close_prices - sma_250_values) / sma_250_values) * 100
        
        # Rate of Change (ROC)
        df['ROC'] = df['Close'].pct_change(periods=self.roc_period) * 100
        
        # Volatility (20-day standard deviation)
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Moving Average Crossover
        df['MA_Crossover'] = df['SMA_50'] - df['SMA_250']
        df['Golden_Cross'] = (df['MA_Crossover'] > 0) & (df['MA_Crossover'].shift(1) <= 0)
        df['Death_Cross'] = (df['MA_Crossover'] < 0) & (df['MA_Crossover'].shift(1) >= 0)
        
        self.data = df.dropna()
        return self.data
    
    def generate_signals(self):
        """Generate signals for switching between TQQQ and SQQQ"""
        df = self.data.copy()
        
        df['Signal'] = 0  # 1 for TQQQ, -1 for SQQQ, 0 for cash
        df['Position'] = 0  # 1 for TQQQ, -1 for SQQQ, 0 for cash
        df['Active_ETF'] = ''  # 'TQQQ' or 'SQQQ'
        df['Entry_Price'] = 0.0
        df['Profit_Target'] = 0.0
        df['Stop_Loss'] = 0.0
        
        in_position = False
        current_etf = None
        
        for i in range(1, len(df)):
            base_price = df['Close'].iloc[i]  # QQQ price for signals
            tqqq_price = df['TQQQ_Close'].iloc[i]
            sqqq_price = df['SQQQ_Close'].iloc[i]
            
            # Not in position - Look for new entry
            if not in_position:
                # Bullish Signal - Enter TQQQ
                if (df['Golden_Cross'].iloc[i] and 
                    base_price > df['SMA_50'].iloc[i] and 
                    base_price > df['SMA_250'].iloc[i] and
                    df['ROC'].iloc[i] > 0):
                    
                    df.loc[df.index[i], 'Signal'] = 1
                    df.loc[df.index[i], 'Position'] = 1
                    df.loc[df.index[i], 'Active_ETF'] = 'TQQQ'
                    df.loc[df.index[i], 'Entry_Price'] = tqqq_price
                    df.loc[df.index[i], 'Profit_Target'] = tqqq_price * (1 + self.profit_target)
                    df.loc[df.index[i], 'Stop_Loss'] = tqqq_price * (1 - self.stop_loss)
                    in_position = True
                    current_etf = 'TQQQ'
                
                # Bearish Signal - Enter SQQQ
                elif (df['Death_Cross'].iloc[i] and 
                      base_price < df['SMA_50'].iloc[i] and 
                      base_price < df['SMA_250'].iloc[i] and
                      df['ROC'].iloc[i] < 0):
                    
                    df.loc[df.index[i], 'Signal'] = -1
                    df.loc[df.index[i], 'Position'] = -1
                    df.loc[df.index[i], 'Active_ETF'] = 'SQQQ'
                    df.loc[df.index[i], 'Entry_Price'] = sqqq_price
                    df.loc[df.index[i], 'Profit_Target'] = sqqq_price * (1 + self.profit_target)
                    df.loc[df.index[i], 'Stop_Loss'] = sqqq_price * (1 - self.stop_loss)
                    in_position = True
                    current_etf = 'SQQQ'
            
            # In position - Check exit conditions
            elif in_position:
                entry_price = df.loc[df.index[i], 'Entry_Price']
                profit_target = df.loc[df.index[i], 'Profit_Target']
                stop_loss = df.loc[df.index[i], 'Stop_Loss']
                current_price = tqqq_price if current_etf == 'TQQQ' else sqqq_price
                
                # Exit conditions
                exit_signal = False
                
                # For TQQQ position
                if current_etf == 'TQQQ':
                    # Calculate distance to 250-day MA
                    distance_to_ma = ((base_price - df['SMA_250'].iloc[i]) / df['SMA_250'].iloc[i] * 100)
                    
                    # Exit if:
                    # 1. Price is within 7% of 250-day MA and trending down
                    # 2. Profit target reached
                    # 3. Stop loss hit
                    if ((distance_to_ma <= 7 and df['ROC'].iloc[i] < 0) or
                        current_price >= profit_target or
                        current_price <= stop_loss):
                        exit_signal = True
                
                # For SQQQ position
                elif current_etf == 'SQQQ':
                    if (df['Golden_Cross'].iloc[i] or
                        current_price >= profit_target or
                        current_price <= stop_loss):
                        exit_signal = True
                
                if exit_signal:
                    df.loc[df.index[i], 'Signal'] = -1 if current_etf == 'TQQQ' else 1
                    df.loc[df.index[i], 'Position'] = 0
                    df.loc[df.index[i], 'Active_ETF'] = ''
                    in_position = False
                    current_etf = None
                else:
                    df.loc[df.index[i], 'Position'] = 1 if current_etf == 'TQQQ' else -1
                    df.loc[df.index[i], 'Active_ETF'] = current_etf
        
        self.data = df
        return self.data
    
    def backtest_strategy(self, initial_capital=100000):
        """Backtest the strategy and calculate returns"""
        df = self.data.copy()
        
        # Calculate base returns
        df['QQQ_Returns'] = df['Close'].pct_change()
        df['TQQQ_Returns'] = df['TQQQ_Close'].pct_change()
        df['SQQQ_Returns'] = df['SQQQ_Close'].pct_change()
        
        # Initialize tracking columns
        df['Position'] = 0  # 1 for TQQQ, -1 for SQQQ, 0 for cash
        df['Portfolio_Value'] = initial_capital
        df['Daily_Returns'] = 0.0
        
        # Track trades
        self.trades = []
        position = None
        entry_price = 0
        entry_date = None
        
        for i in range(1, len(df)):
            current_date = df.index[i]
            prev_date = df.index[i-1]
            
            # Calculate daily return based on position
            if df.loc[prev_date, 'Position'] == 1:  # TQQQ position
                df.loc[current_date, 'Daily_Returns'] = df.loc[current_date, 'TQQQ_Returns']
            elif df.loc[prev_date, 'Position'] == -1:  # SQQQ position
                df.loc[current_date, 'Daily_Returns'] = df.loc[current_date, 'SQQQ_Returns']
            
            # Update portfolio value
            df.loc[current_date, 'Portfolio_Value'] = (
                df.loc[prev_date, 'Portfolio_Value'] * 
                (1 + df.loc[current_date, 'Daily_Returns'])
            )
            
            # Handle signals
            if df.loc[current_date, 'Signal'] != 0:
                if position is None:  # Enter new position
                    position = 'TQQQ' if df.loc[current_date, 'Signal'] == 1 else 'SQQQ'
                    entry_price = df.loc[current_date, f'{position}_Close']
                    entry_date = current_date
                    df.loc[current_date:, 'Position'] = 1 if position == 'TQQQ' else -1
                else:  # Exit position
                    exit_price = df.loc[current_date, f'{position}_Close']
                    
                    # Record trade
                    self.trades.append({
                        'Entry_Date': entry_date,
                        'Exit_Date': current_date,
                        'Position': position,
                        'Entry_Price': entry_price,
                        'Exit_Price': exit_price,
                        'Profit': (exit_price - entry_price) / entry_price * 100,
                        'Holding_Days': (current_date - entry_date).days
                    })
                    
                    position = None
                    df.loc[current_date:, 'Position'] = 0
        
        # Calculate cumulative returns
        df['Strategy_Returns'] = df['Daily_Returns']
        df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
        df['Cumulative_QQQ'] = (1 + df['QQQ_Returns']).cumprod()
        
        self.data = df
        return df
        
        # Track individual trades
        self.trades = []
        current_position = None
        entry_price = 0
        entry_date = None
        shares = 0
        running_capital = initial_capital
        
        for i in range(1, len(df)):
            current_date = df.index[i]
            
            # Handle position entry
            if df['Signal'].iloc[i] != 0 and current_position is None:
                current_position = 'TQQQ' if df['Signal'].iloc[i] == 1 else 'SQQQ'
                entry_price = df[f'{current_position}_Close'].iloc[i]
                entry_date = current_date
                shares = running_capital * 0.95 / entry_price  # Use 95% of capital for position
                df.loc[current_date, 'Cash'] = running_capital * 0.05  # Keep 5% as cash
                df.loc[current_date, 'Position_Value'] = shares * entry_price
                
            # Handle position exit
            elif df['Signal'].iloc[i] != 0 and current_position is not None:
                exit_price = df[f'{current_position}_Close'].iloc[i]
                exit_date = current_date
                position_value = shares * exit_price
                
                # Calculate trade metrics
                trade_return = ((exit_price - entry_price) / entry_price) * 100
                if current_position == 'SQQQ':  # Invert return for SQQQ positions
                    trade_return = -trade_return
                    
                trade_profit = position_value - (shares * entry_price)
                running_capital = position_value + df['Cash'].iloc[i-1]
                
                # Record trade details
                self.trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': exit_date,
                    'Position': current_position,
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Shares': shares,
                    'Initial_Value': shares * entry_price,
                    'Final_Value': position_value,
                    'Profit': trade_profit,
                    'Return_%': trade_return,
                    'Holding_Days': (exit_date - entry_date).days,
                    'Exit_Reason': 'Profit Target' if exit_price >= df['Profit_Target'].iloc[i] else 
                                 'Stop Loss' if exit_price <= df['Stop_Loss'].iloc[i] else 'Signal Change',
                    'Portfolio_Value': running_capital
                })
                
                # Update portfolio tracking
                df.loc[current_date, 'Cash'] = running_capital
                df.loc[current_date, 'Position_Value'] = 0
                df.loc[current_date, 'Trade_Returns'] = trade_return
                current_position = None
                shares = 0
                
            # Update daily portfolio value
            if current_position is not None:
                current_price = df[f'{current_position}_Close'].iloc[i]
                df.loc[current_date, 'Position_Value'] = shares * current_price
            else:
                df.loc[current_date, 'Position_Value'] = 0.0
                
            df.loc[current_date, 'Portfolio_Value'] = (
                df.loc[current_date, 'Cash'] + df.loc[current_date, 'Position_Value']
            )
            
            if i > 0:
                prev_value = df['Portfolio_Value'].iloc[i-1]
                if prev_value > 0:  # Avoid division by zero
                    df.loc[current_date, 'Strategy_Returns'] = (
                        df.loc[current_date, 'Portfolio_Value'] / prev_value - 1
                    )
        
        # Calculate cumulative returns
        df['Strategy_Returns'] = df['Strategy_Returns'].fillna(0)
        df['Returns'] = df['Returns'].fillna(0)
        
        df['Cumulative_Market_Returns'] = (1 + df['Returns']).cumprod()
        df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()
        
        self.data = df
        return df
        
        self.data = df
        return df
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        df = self.data
        
        # Filter out NaN and inf values
        df['Strategy_Returns'] = df['Strategy_Returns'].replace([np.inf, -np.inf], np.nan)
        df['QQQ_Returns'] = df['QQQ_Returns'].replace([np.inf, -np.inf], np.nan)
        
        # Calculate total returns
        total_strategy_return = (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0] - 1) * 100
        total_market_return = (df['Cumulative_QQQ'].iloc[-1] - 1) * 100
        
        # Calculate annualized returns
        years = (df.index[-1] - df.index[0]).days / 365.25
        annualized_strategy = ((1 + total_strategy_return/100) ** (1/years) - 1) * 100
        annualized_market = ((1 + total_market_return/100) ** (1/years) - 1) * 100
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # 2% risk-free rate
        excess_returns = df['Strategy_Returns'].dropna() - risk_free_rate/252
        strategy_sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(excess_returns) > 0 else 0
        
        excess_returns_mkt = df['QQQ_Returns'].dropna() - risk_free_rate/252
        market_sharpe = np.sqrt(252) * excess_returns_mkt.mean() / excess_returns_mkt.std() if len(excess_returns_mkt) > 0 else 0
        
        cumulative = df['Cumulative_Strategy_Returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        market_cumulative = df['Cumulative_Market_Returns']
        market_running_max = market_cumulative.expanding().max()
        market_drawdown = (market_cumulative - market_running_max) / market_running_max
        market_max_drawdown = market_drawdown.min() * 100
        
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            win_rate = (trades_df['Return_%'] > 0).sum() / len(trades_df) * 100
            avg_return = trades_df['Return_%'].mean()
            avg_holding = trades_df['Holding_Days'].mean()
            best_trade = trades_df['Return_%'].max()
            worst_trade = trades_df['Return_%'].min()
            avg_winning = trades_df[trades_df['Return_%'] > 0]['Return_%'].mean()
            avg_losing = trades_df[trades_df['Return_%'] < 0]['Return_%'].mean()
        else:
            win_rate = avg_return = avg_holding = 0
            best_trade = worst_trade = avg_winning = avg_losing = 0
        
        metrics = {
            'Backtest_Period_Years': round(years, 2),
            'Total_Strategy_Return_%': round(total_strategy_return, 2),
            'Total_Market_Return_%': round(total_market_return, 2),
            'Annualized_Strategy_Return_%': round(annualized_strategy, 2),
            'Annualized_Market_Return_%': round(annualized_market, 2),
            'Outperformance_%': round(total_strategy_return - total_market_return, 2),
            'Strategy_Sharpe_Ratio': round(strategy_sharpe, 2),
            'Market_Sharpe_Ratio': round(market_sharpe, 2),
            'Max_Drawdown_%': round(max_drawdown, 2),
            'Market_Max_Drawdown_%': round(market_max_drawdown, 2),
            'Number_of_Trades': len(self.trades),
            'Win_Rate_%': round(win_rate, 2),
            'Avg_Return_per_Trade_%': round(avg_return, 2),
            'Best_Trade_%': round(best_trade, 2) if best_trade != 0 else 0,
            'Worst_Trade_%': round(worst_trade, 2) if worst_trade != 0 else 0,
            'Avg_Winning_Trade_%': round(avg_winning, 2) if avg_winning == avg_winning else 0,
            'Avg_Losing_Trade_%': round(avg_losing, 2) if avg_losing == avg_losing else 0,
            'Avg_Holding_Period_Days': round(avg_holding, 2)
        }
        
        return metrics
    
    def run_strategy(self, start_date=None, end_date=None, initial_capital=100000):
        """Execute the complete strategy"""
        self.fetch_data(start_date, end_date)
        self.calculate_indicators()
        self.generate_signals()
        self.backtest_strategy(initial_capital)
        metrics = self.calculate_metrics()
        
        return self.data, metrics, self.trades


# Streamlit UI
def main():
    st.title("📈 50/250-Day Dual Moving Average Strategy")
    st.markdown("---")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("⚙️ Strategy Parameters")
        
        st.info("Using QQQ for signal generation, trading TQQQ/SQQQ for leveraged returns")
        base_ticker = "QQQ"  # Fixed to QQQ
        
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365*10))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        st.subheader("Trading Parameters")
        fast_ma = st.slider("Fast Moving Average (days)", 10, 100, 50, 5)
        slow_ma = st.slider("Slow Moving Average (days)", 100, 500, 250, 10)
        profit_target = st.slider("Profit Target (%)", 10, 50, 15, 5) / 100
        stop_loss = st.slider("Stop Loss (%)", 5, 30, 10, 5) / 100
        roc_period = st.slider("Rate of Change Period (days)", 5, 50, 20, 5)
        
        st.subheader("Portfolio")
        initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000)
        
        st.warning("⚠️ Warning: TQQQ and SQQQ are 3x leveraged ETFs. They carry higher risks and may not be suitable for all investors.")
        
        run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    # Main content
    if run_button:
        with st.spinner(f"Fetching data and running strategy..."):
            try:
                # Initialize and run strategy
                strategy = LeveragedETFStrategy(
                    base_ticker=base_ticker,
                    fast_ma=fast_ma,
                    slow_ma=slow_ma,
                    profit_target=profit_target,
                    stop_loss=stop_loss,
                    roc_period=roc_period
                )
                
                data, metrics, trades = strategy.run_strategy(
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    initial_capital=initial_capital
                )
                
                # Display key metrics
                st.header("📊 Performance Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{metrics['Total_Strategy_Return_%']}%", 
                             delta=f"{metrics['Outperformance_%']}% vs Market")
                with col2:
                    st.metric("Annualized Return", f"{metrics['Annualized_Strategy_Return_%']}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{metrics['Strategy_Sharpe_Ratio']}")
                with col4:
                    st.metric("Max Drawdown", f"{metrics['Max_Drawdown_%']}%")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trades", f"{metrics['Number_of_Trades']}")
                with col2:
                    st.metric("Win Rate", f"{metrics['Win_Rate_%']}%")
                with col3:
                    st.metric("Avg Return/Trade", f"{metrics['Avg_Return_per_Trade_%']}%")
                with col4:
                    st.metric("Avg Holding Period", f"{metrics['Avg_Holding_Period_Days']:.0f} days")
                
                # Create interactive charts
                st.header("📈 Interactive Charts")
                
                # Chart 1: Price vs MA with signals
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=data.index, y=data['Close'], name='QQQ Price', 
                                         line=dict(color='blue', width=2)))
                fig1.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='50-day SMA',
                                         line=dict(color='orange', width=2)))
                fig1.add_trace(go.Scatter(x=data.index, y=data['SMA_250'], name='250-day SMA',
                                         line=dict(color='purple', width=2)))
                
                # TQQQ entry points
                tqqq_entries = data[data['Signal'] == 1]
                tqqq_exits = data[(data['Signal'] == -1) & (data['Active_ETF'].shift(1) == 'TQQQ')]
                
                # SQQQ entry points
                sqqq_entries = data[data['Signal'] == -1]
                sqqq_exits = data[(data['Signal'] == 1) & (data['Active_ETF'].shift(1) == 'SQQQ')]
                
                fig1.add_trace(go.Scatter(x=tqqq_entries.index, y=tqqq_entries['Close'],
                                         mode='markers', name='Enter TQQQ',
                                         marker=dict(color='green', size=12, symbol='triangle-up')))
                fig1.add_trace(go.Scatter(x=tqqq_exits.index, y=tqqq_exits['Close'],
                                         mode='markers', name='Exit TQQQ',
                                         marker=dict(color='red', size=12, symbol='triangle-down')))
                                         
                fig1.add_trace(go.Scatter(x=sqqq_entries.index, y=sqqq_entries['Close'],
                                         mode='markers', name='Enter SQQQ',
                                         marker=dict(color='orange', size=12, symbol='triangle-down')))
                fig1.add_trace(go.Scatter(x=sqqq_exits.index, y=sqqq_exits['Close'],
                                         mode='markers', name='Exit SQQQ',
                                         marker=dict(color='lime', size=12, symbol='triangle-up')))
                
                fig1.update_layout(title='QQQ Price with TQQQ/SQQQ Signals',
                                  xaxis_title='Date', yaxis_title='Price ($)',
                                  height=500, hovermode='x unified')
                fig1.update_yaxis(type='log')
                st.plotly_chart(fig1, use_container_width=True)
                
                # Chart 2: Distance from MA
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=data.index, y=data['Distance_from_Slow_MA']*100,
                                         name='Distance from 250-day MA', fill='tozeroy',
                                         line=dict(color='blue', width=2)))
                fig2.add_trace(go.Scatter(x=data.index, y=data['Distance_from_Fast_MA']*100,
                                         name='Distance from 50-day MA',
                                         line=dict(color='orange', width=2)))
                
                # Add horizontal lines for the 7% threshold
                fig2.add_hline(y=7, line_dash="dash", line_color="red",
                              annotation_text="Exit Threshold (7%)")
                fig2.add_hline(y=-7, line_dash="dash", line_color="green",
                              annotation_text="Entry Threshold (-7%)")
                fig2.add_hline(y=0, line_color="black", line_width=1)
                
                fig2.update_layout(title='Distance from Moving Averages',
                                  xaxis_title='Date', yaxis_title='Distance (%)',
                                  height=400, hovermode='x unified')
                st.plotly_chart(fig2, use_container_width=True)
                
                # Chart 3: Cumulative Returns
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=data.index, y=(data['Cumulative_QQQ']-1)*100,
                                         name='QQQ Buy & Hold', line=dict(color='gray', width=3)))
                fig3.add_trace(go.Scatter(x=data.index, y=(data['Portfolio_Value']/data['Portfolio_Value'].iloc[0]-1)*100,
                                         name='TQQQ/SQQQ Strategy', line=dict(color='green', width=3)))
                
                fig3.update_layout(title='Strategy Performance vs Buy & Hold',
                                  xaxis_title='Date', yaxis_title='Cumulative Returns (%)',
                                  height=500, hovermode='x unified')
                st.plotly_chart(fig3, use_container_width=True)
                
                # Chart 4: Drawdown Analysis
                cumulative = data['Cumulative_Strategy_Returns']
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max * 100
                
                market_cumulative = data['Cumulative_Market_Returns']
                market_running_max = market_cumulative.expanding().max()
                market_drawdown = (market_cumulative - market_running_max) / market_running_max * 100
                
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=data.index, y=drawdown, name='Strategy Drawdown',
                                         fill='tozeroy', line=dict(color='red', width=2)))
                fig4.add_trace(go.Scatter(x=data.index, y=market_drawdown, name='Market Drawdown',
                                         line=dict(color='blue', width=2)))
                
                fig4.update_layout(title='Drawdown Analysis',
                                  xaxis_title='Date', yaxis_title='Drawdown (%)',
                                  height=400, hovermode='x unified')
                st.plotly_chart(fig4, use_container_width=True)
                
                # Detailed Metrics
                st.header("📋 Detailed Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("📊 Return Analysis")
                    st.write(f"**Backtest Period:** {metrics['Backtest_Period_Years']} years")
                    st.write(f"**Total Strategy Return:** {metrics['Total_Strategy_Return_%']}%")
                    st.write(f"**Total Market Return:** {metrics['Total_Market_Return_%']}%")
                    st.write(f"**Annualized Strategy:** {metrics['Annualized_Strategy_Return_%']}%")
                    st.write(f"**Annualized Market:** {metrics['Annualized_Market_Return_%']}%")
                    st.write(f"**Outperformance:** {metrics['Outperformance_%']}%")
                
                with col2:
                    st.subheader("📉 Risk Analysis")
                    st.write(f"**Strategy Sharpe:** {metrics['Strategy_Sharpe_Ratio']}")
                    st.write(f"**Market Sharpe:** {metrics['Market_Sharpe_Ratio']}")
                    st.write(f"**Strategy Max DD:** {metrics['Max_Drawdown_%']}%")
                    st.write(f"**Market Max DD:** {metrics['Market_Max_Drawdown_%']}%")
                    
                    final_value = initial_capital * data['Cumulative_Strategy_Returns'].iloc[-1]
                    market_value = initial_capital * data['Cumulative_Market_Returns'].iloc[-1]
                    st.write(f"**Final Portfolio:** ${final_value:,.2f}")
                    st.write(f"**Buy & Hold:** ${market_value:,.2f}")
                
                with col3:
                    st.subheader("📈 Trade Analysis")
                    st.write(f"**Total Trades:** {metrics['Number_of_Trades']}")
                    st.write(f"**Win Rate:** {metrics['Win_Rate_%']}%")
                    st.write(f"**Avg Return/Trade:** {metrics['Avg_Return_per_Trade_%']}%")
                    st.write(f"**Best Trade:** {metrics['Best_Trade_%']}%")
                    st.write(f"**Worst Trade:** {metrics['Worst_Trade_%']}%")
                    st.write(f"**Avg Winning:** {metrics['Avg_Winning_Trade_%']}%")
                    st.write(f"**Avg Losing:** {metrics['Avg_Losing_Trade_%']}%")
                    st.write(f"**Avg Holding:** {metrics['Avg_Holding_Period_Days']:.0f} days")
                
                # Detailed Trade Analysis
                if len(trades) > 0:
                    st.header("� Detailed Trade Analysis")
                    trades_df = pd.DataFrame(trades)
                    
                    # Format dates
                    trades_df['Entry_Date'] = pd.to_datetime(trades_df['Entry_Date'])
                    trades_df['Exit_Date'] = pd.to_datetime(trades_df['Exit_Date'])
                    trades_df['Year'] = trades_df['Entry_Date'].dt.year
                    trades_df['Quarter'] = trades_df['Entry_Date'].dt.quarter
                    
                    # Portfolio Growth Analysis
                    st.subheader("💰 Portfolio Growth Analysis")
                    initial_value = 10000  # Fixed initial capital for comparison
                    current_value = initial_value
                    portfolio_track = []
                    
                    for _, trade in trades_df.iterrows():
                        portfolio_track.append({
                            'Date': trade['Entry_Date'],
                            'Action': f"Enter {trade['Position']}",
                            'Portfolio_Value': current_value,
                            'Trade_Value': trade['Initial_Value']
                        })
                        portfolio_track.append({
                            'Date': trade['Exit_Date'],
                            'Action': f"Exit {trade['Position']}",
                            'Portfolio_Value': trade['Portfolio_Value'],
                            'Profit': trade['Profit']
                        })
                        current_value = trade['Portfolio_Value']
                    
                    portfolio_df = pd.DataFrame(portfolio_track)
                    
                    # Display portfolio growth
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_return = ((current_value - initial_value) / initial_value) * 100
                        st.metric("Total Return", f"{total_return:.2f}%", 
                                f"${current_value - initial_value:,.2f}")
                    with col2:
                        avg_trade_profit = trades_df['Profit'].mean()
                        st.metric("Average Profit per Trade", f"${avg_trade_profit:,.2f}")
                    with col3:
                        win_rate = (trades_df['Profit'] > 0).mean() * 100
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    
                    # Quarterly Performance
                    st.subheader("📈 Quarterly Performance")
                    quarterly = trades_df.groupby(['Year', 'Quarter']).agg({
                        'Profit': ['sum', 'count'],
                        'Return_%': 'mean',
                        'Holding_Days': 'mean'
                    }).round(2)
                    quarterly.columns = ['Total Profit', 'Trades', 'Avg Return %', 'Avg Days']
                    st.dataframe(quarterly, use_container_width=True)
                    
                    # Detailed Trade List
                    st.subheader("🔍 Detailed Trade List")
                    detailed_trades = trades_df[[
                        'Entry_Date', 'Position', 'Entry_Price', 'Exit_Date', 
                        'Exit_Price', 'Shares', 'Profit', 'Return_%', 
                        'Holding_Days', 'Exit_Reason'
                    ]].copy()
                    
                    # Format columns
                    detailed_trades['Entry_Date'] = detailed_trades['Entry_Date'].dt.strftime('%Y-%m-%d')
                    detailed_trades['Exit_Date'] = detailed_trades['Exit_Date'].dt.strftime('%Y-%m-%d')
                    detailed_trades['Entry_Price'] = detailed_trades['Entry_Price'].round(2)
                    detailed_trades['Exit_Price'] = detailed_trades['Exit_Price'].round(2)
                    detailed_trades['Shares'] = detailed_trades['Shares'].round(2)
                    detailed_trades['Profit'] = detailed_trades['Profit'].round(2)
                    detailed_trades['Return_%'] = detailed_trades['Return_%'].round(2)
                    
                    st.dataframe(detailed_trades, use_container_width=True)
                    
                    # Trade Statistics
                    st.subheader("📊 Trade Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Position Analysis**")
                        position_stats = trades_df.groupby('Position').agg({
                            'Profit': ['count', 'mean', 'sum'],
                            'Return_%': 'mean',
                            'Holding_Days': 'mean'
                        }).round(2)
                        position_stats.columns = ['Count', 'Avg Profit', 'Total Profit', 
                                               'Avg Return %', 'Avg Days']
                        st.dataframe(position_stats)
                        
                    with col2:
                        st.write("**Exit Reason Analysis**")
                        exit_stats = trades_df.groupby('Exit_Reason').agg({
                            'Profit': ['count', 'mean', 'sum'],
                            'Return_%': 'mean'
                        }).round(2)
                        exit_stats.columns = ['Count', 'Avg Profit', 'Total Profit', 'Avg Return %']
                        st.dataframe(exit_stats)
                    
                    # Download button
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Trade History (CSV)",
                        data=csv,
                        file_name=f"{ticker}_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                # Current Status
                st.header("🎯 Current Market Status")
                latest = data.iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("QQQ Price", f"${latest['Close']:.2f}")
                    st.metric("TQQQ Price", f"${latest['TQQQ_Close']:.2f}")
                    st.metric("SQQQ Price", f"${latest['SQQQ_Close']:.2f}")
                with col2:
                    st.metric("50-day MA", f"${latest['SMA_50']:.2f}")
                    st.metric("250-day MA", f"${latest['SMA_250']:.2f}")
                    st.metric("ROC", f"{latest['ROC']:.2f}%")
                with col3:
                    if latest['Position'] == 1:
                        position_status = "🟢 LONG (TQQQ)"
                    elif latest['Position'] == -1:
                        position_status = "🔴 SHORT (SQQQ)"
                    else:
                        position_status = "💵 CASH"
                    st.metric("Current Position", position_status)
                    if latest['Position'] != 0:
                        st.metric("Entry Price", f"${latest['Entry_Price']:.2f}")
                        st.metric("Profit Target", f"${latest['Profit_Target']:.2f}")
                
                if latest['Position'] == 0:
                    slow_dist = ((latest['Close'] - latest['SMA_250']) / latest['SMA_250'] * 100)
                    fast_dist = ((latest['Close'] - latest['SMA_50']) / latest['SMA_50'] * 100)
                    
                    if slow_dist > 7 and fast_dist > 7:
                        st.warning(f"⚠️ POTENTIAL SQQQ OPPORTUNITY - Price is {slow_dist:.1f}% above 250-day MA")
                    elif slow_dist < -7 and fast_dist < -7:
                        st.warning(f"⚠️ POTENTIAL TQQQ OPPORTUNITY - Price is {abs(slow_dist):.1f}% below 250-day MA")
                    else:
                        st.success("✅ Wait for clearer trend signals")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please check the ticker symbol and date range.")
    
    else:
        # Display instructions
        st.info("👈 Configure strategy parameters in the sidebar and click 'Run Backtest' to start")
        
        st.markdown("""
        ## How It Works
        
        This leveraged ETF strategy uses TQQQ (3x Long) and SQQQ (3x Short) to amplify returns based on QQQ's trend:
        
        ### 📊 Bullish Signal (Buy TQQQ)
        - **Golden Cross**: 50-day MA crosses above 250-day MA
        - QQQ price is above both moving averages
        - Positive Rate of Change (upward momentum)
        
        ### 📉 Bearish Signal (Buy SQQQ)
        - **Death Cross**: 50-day MA crosses below 250-day MA
        - QQQ price is below both moving averages
        - Negative Rate of Change (downward momentum)
        
        ### 🎯 Exit Signals
        For TQQQ (Long Position):
        - Price approaches within 7% of 250-day MA during downtrend
        - Profit target reached (default 15%)
        - Stop loss hit (default 10%)
        
        For SQQQ (Short Position):
        - Golden Cross (50-day MA crosses above 250-day MA)
        - Profit target reached (default 15%)
        - Stop loss hit (default 10%)
        
        ### ⚠️ Risk Warning
        - TQQQ and SQQQ are 3x leveraged ETFs
        - They carry higher risks due to daily rebalancing
        - Not suitable for long-term holding
        - Use strict position sizing and risk management
        
        ### 🔍 Key Features
        - **20-year backtesting** with real historical data
        - **Interactive charts** powered by Plotly
        - **Comprehensive metrics** including Sharpe ratio, max drawdown, win rate
        - **Trade history** with yearly breakdown
        - **Real-time signals** showing current market status
        
        ### 💡 Tips
        - Test different tickers (SPY, QQQ, AAPL, etc.)
        - Adjust entry/exit thresholds based on volatility
        - Longer MA periods work better for long-term strategies
        - Review drawdown periods to understand risk exposure
        """)


if __name__ == "__main__":
    main()