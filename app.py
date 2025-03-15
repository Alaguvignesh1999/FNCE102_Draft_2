import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import openpyxl
import statsmodels.api as sm
import numpy as np
import io

st.title("Portfolio vs SP500 Analysis")
st.write("""
This Streamlit app allows you to analyze the portfolio performance relative to SP500.
You can view metrics like Sharpe ratio, Jensen's Alpha, Drawdowns, etc.
""")

#Download daily data for SP500 for 2009-2024
start_date = '2009-01-01'
end_date = '2024-12-31'
sp500_full = yf.download('SPY', start=start_date, end=end_date)
sp500_dailyclose = sp500_full['Close']

#Convert SP500 to daily returns
sp500_dailyreturn = sp500_dailyclose.pct_change()

#Calculate monthly returns and volatility by first, resampling daily returns to monthly returns, then calculating monthly volatility
sp500_monthlyreturn = sp500_dailyreturn.resample('ME').agg(lambda x: (x+1).prod() - 1)

#Read monthly rebalancing schedule

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload your rebalancing schedule (Excel file)", type=["xlsx", "xls"])

# Add a slider for the user to select transaction cost
tx_cost = st.slider('Select Transaction Cost (Up to 2% of urnover volume)', 0.0, 2.0, 0.1) * 0.01  # Min: 0, Max: 0.2, Default: 0.0001


if uploaded_file is not None:
    # Read the uploaded Excel file
    rebalance_data = pd.read_excel(uploaded_file, index_col='Date', parse_dates=True)
    st.write("Rebalancing schedule successfully loaded!")

    # Step 1: Get all unique tickers from rebalance_data
    unique_tickers = set(rebalance_data.columns)

    # Step 2: Fetch historical data for all tickers at once (reduce API calls)
    calc_start_date = rebalance_data.index.min() # Earliest rebalance date
    calc_end_date = rebalance_data.index.max() + pd.offsets.MonthEnd(0)  # Ensure full coverage

    # Download all tickers in one go
    data = yf.download(list(unique_tickers), start=calc_start_date, end=calc_end_date, progress=False)['Close']

    # Resample monthly returns once
    calc_monthly_returns = data.pct_change().resample('ME').agg(lambda x: (x + 1).prod() - 1)


    # Initialize DataFrame with Date, Returns, Turnover %, and each ticker
    portfolio_returns = pd.DataFrame(columns=['Date', 'Returns', 'Turnover %'] + list(rebalance_data.columns))

    # Iterate through rebalance data for actual monthly updates
    for date, row in rebalance_data.iterrows():
        port_monthly_returns = 0
        last_day_of_month = date + pd.offsets.MonthEnd(0)  # Convert to last day of the month
        
        # Store the previous weights from the last recorded row
        prev_weights = portfolio_returns.iloc[-1, 3:].astype(float) if not portfolio_returns.empty else row

        new_weights = {}
        for ticker, weight in row.items():
            if weight > 0:
                new_weights[ticker] = weight * (1 + calc_monthly_returns.loc[last_day_of_month, ticker])
                port_monthly_returns += calc_monthly_returns.loc[last_day_of_month, ticker] * weight
            else:
                new_weights[ticker] = 0  # If not in portfolio, set to 0

        # Compute turnover as the sum of absolute changes in weight
        turnover = sum(abs(new_weights.get(ticker, 0) - prev_weights.get(ticker, 0)) for ticker in rebalance_data.columns)

        # Append new row
        new_row = pd.DataFrame([[last_day_of_month.floor("D"), port_monthly_returns, turnover] + list(new_weights.values())],
                            columns=portfolio_returns.columns)
        portfolio_returns = pd.concat([portfolio_returns, new_row], ignore_index=True)

    # Add first-of-month row (2009-01-01) after the loop
    first_date = rebalance_data.index.min()
    first_weights = rebalance_data.iloc[0]  # First set of weights

    first_row = pd.DataFrame([[first_date, 0.0, 1.0] + list(first_weights)], 
                            columns=portfolio_returns.columns)

    # Add it at the beginning of the DataFrame
    portfolio_returns = pd.concat([first_row, portfolio_returns], ignore_index=True)

    # Convert 'Date' to datetime type
    portfolio_returns['Date'] = pd.to_datetime(portfolio_returns['Date'])

    # Set 'Date' as index
    portfolio_returns.set_index('Date', inplace=True)

    # Fill NaNs with 0 for missing tickers in certain months
    portfolio_returns.fillna(0, inplace=True)

    # Set initial portfolio value
    initial_value = 10_000_000  # 10 million
    # tx_cost = 0.0001

    # Create a new column 'Value' and initialize with NaN
    portfolio_returns['Portfolio Wealth Curve'] = None
    print(tx_cost)
    # Set the first month's value
    portfolio_returns.iloc[0, portfolio_returns.columns.get_loc('Portfolio Wealth Curve')] = initial_value - initial_value * portfolio_returns.loc[portfolio_returns.index[0],'Turnover %'] * tx_cost

    # Loop through subsequent months and update the value
    for i in range(1, len(portfolio_returns)):
        previous_value = portfolio_returns.iloc[i - 1]['Portfolio Wealth Curve']
        current_return = portfolio_returns.iloc[i]['Returns']
        portfolio_returns.iloc[i, portfolio_returns.columns.get_loc('Portfolio Wealth Curve')] = previous_value * (1 + current_return) - previous_value * portfolio_returns.iloc[i]['Turnover %'] * tx_cost

    # Convert the 'Value' column to numeric type
    portfolio_returns['Portfolio Wealth Curve'] = portfolio_returns['Portfolio Wealth Curve'].astype(float)








    output = pd.merge(sp500_monthlyreturn, portfolio_returns, left_index = True, right_index = True, how = 'outer')
    output.fillna(0,inplace=True)

    # Check if 'SPY_y' exists before renaming
    if 'SPY_x' in output.columns:
        output.rename(columns={'SPY_x': 'SPY'}, inplace=True)
    if 'SPY_y' in output.columns:
        output.rename(columns={'SPY_y': 'SPY_Portfolio'}, inplace=True)


    # Set initial portfolio value
    initial_value = 10_000_000  # 10 million

    #Create a bew column 'Portfolio Wealth Curve' and initialize with NaN
    output['SPY Wealth Curve'] = None
    output['Portfolio Wealth Curve'] = None

    # Set the first month's value
    output.iloc[0, output.columns.get_loc('SPY Wealth Curve')] = initial_value - initial_value * output.loc[output.index[0],'Turnover %'] * tx_cost
    output.iloc[0, output.columns.get_loc('Portfolio Wealth Curve')] = initial_value - initial_value * output.loc[output.index[0],'Turnover %'] * tx_cost

    for i in range(1, len(output)):
        #Loop through SPY and update the value
        previous_value_spy = output.iloc[i - 1]['SPY Wealth Curve']
        print(output.columns)
        current_return_spy = output.iloc[i]['SPY']
        output.iloc[i, output.columns.get_loc('SPY Wealth Curve')] = previous_value_spy * (1 + current_return_spy)


        #Loop through portfolio returns and update the value
        previous_value_port = output.iloc[i - 1]['Portfolio Wealth Curve']
        current_return_port = output.iloc[i]['Returns']
        output.iloc[i, output.columns.get_loc('Portfolio Wealth Curve')] = previous_value_port * (1 + current_return_port) - previous_value_port * output.iloc[i]['Turnover %'] * tx_cost



    month_end_returns = output.copy()

    # Keep only the desired columns
    month_end_returns = month_end_returns[['Turnover %', 'Portfolio Wealth Curve', 'SPY Wealth Curve']]

    # Remove the first row
    month_end_returns = month_end_returns.drop(month_end_returns.index[0])






    # Define the initial portfolio and SPY value (10 million)
    initial_value = 10_000_000

    # Initialize 'Returns' column for Portfolio and SPY
    month_end_returns['SPY Returns'] = None
    month_end_returns['Portfolio Returns'] = None


    # Calculate returns for the first row
    month_end_returns.iloc[0, month_end_returns.columns.get_loc('Portfolio Returns')] = \
        (month_end_returns.iloc[0]['Portfolio Wealth Curve'] - initial_value) / initial_value

    month_end_returns.iloc[0, month_end_returns.columns.get_loc('SPY Returns')] = \
        (month_end_returns.iloc[0]['SPY Wealth Curve'] - initial_value) / initial_value

    # Calculate returns for the rest of the rows
    for i in range(1, len(month_end_returns)):
        # Portfolio returns
        month_end_returns.iloc[i, month_end_returns.columns.get_loc('Portfolio Returns')] = \
            (month_end_returns.iloc[i]['Portfolio Wealth Curve'] - month_end_returns.iloc[i - 1]['Portfolio Wealth Curve']) / \
            month_end_returns.iloc[i - 1]['Portfolio Wealth Curve']
        
        # SPY returns
        month_end_returns.iloc[i, month_end_returns.columns.get_loc('SPY Returns')] = \
            (month_end_returns.iloc[i]['SPY Wealth Curve'] - month_end_returns.iloc[i - 1]['SPY Wealth Curve']) / \
            month_end_returns.iloc[i - 1]['SPY Wealth Curve']



    # Get risk-free rate (1-month Treasury rate)
    rf_data = yf.download("^IRX", start=start_date, end=end_date)  
    rf_monthly = rf_data['Close'].resample('ME').mean() / 100  # Convert % to decimal

    # Convert annualized to monthly rate
    rf_monthly = (1 + rf_monthly) ** (1/12) - 1


    month_end_returns['Rf Rate'] = rf_monthly['^IRX']  # Ensure alignment with month_end_returns
    month_end_returns['SPY Excess'] = month_end_returns['SPY Returns'] - month_end_returns['Rf Rate']
    month_end_returns['Portfolio Excess'] = month_end_returns['Portfolio Returns'] - month_end_returns['Rf Rate']
    month_end_returns['Active Return'] = month_end_returns['Portfolio Returns'] - month_end_returns['SPY Returns']


    # Calculate prior peaks for both portfolio and SPY
    month_end_returns['prior_peaks_portfolio'] = month_end_returns['Portfolio Wealth Curve'].cummax()
    month_end_returns['prior_peaks_spy'] = month_end_returns['SPY Wealth Curve'].cummax()

    # Calculate the drawdowns for portfolio and SPY
    month_end_returns['drawdown_portfolio'] = (month_end_returns['Portfolio Wealth Curve'] - month_end_returns['prior_peaks_portfolio']) / month_end_returns['prior_peaks_portfolio']
    month_end_returns['drawdown_spy'] = (month_end_returns['SPY Wealth Curve'] - month_end_returns['prior_peaks_spy']) / month_end_returns['prior_peaks_spy']


    # Ensure the Date column is a datetime index
    port_full_stats = month_end_returns.copy()

    # Add a 'Year' column
    port_full_stats['Year'] = port_full_stats.index.year

    # Group by Year and compute statistics
    annual_stats = port_full_stats.groupby('Year').agg({
        'SPY Excess': ['mean', 'std'],
        'Portfolio Excess': ['mean', 'std']
    })

    # Convert multi-level columns
    annual_stats.columns = ['SPY_Mean_Excess', 'SPY_Std_Excess', 'Portfolio_Mean_Excess', 'Portfolio_Std_Excess']

    # Annualize the statistics
    annual_stats['SPY_Annualized_Mean'] = annual_stats['SPY_Mean_Excess'] * 12
    annual_stats['Portfolio_Annualized_Mean'] = annual_stats['Portfolio_Mean_Excess'] * 12
    annual_stats['SPY_Annualized_Volatility'] = annual_stats['SPY_Std_Excess'] * np.sqrt(12)
    annual_stats['Portfolio_Annualized_Volatility'] = annual_stats['Portfolio_Std_Excess'] * np.sqrt(12)

    # Calculate Sharpe Ratios
    annual_stats['SPY_Sharpe'] = annual_stats['SPY_Annualized_Mean'] / annual_stats['SPY_Annualized_Volatility']
    annual_stats['Portfolio_Sharpe'] = annual_stats['Portfolio_Annualized_Mean'] / annual_stats['Portfolio_Annualized_Volatility']

    # Reset index to make 'Year' a column
    annual_stats.reset_index(inplace=True)

    # Create a new DataFrame for overall monthly returns
    overall_returns = pd.DataFrame({
        'SP500': port_full_stats['SPY Excess'],
        'Portfolio': port_full_stats['Portfolio Excess']
    })

    # Calculate average monthly returns for SPY and Portfolio
    avg_monthly_return_spy = overall_returns['SP500'].mean()
    avg_monthly_return_portfolio = overall_returns['Portfolio'].mean()

    # Annualize the return
    annualized_return_spy = (1 + avg_monthly_return_spy) ** 12 - 1
    annualized_return_portfolio = (1 + avg_monthly_return_portfolio) ** 12 - 1

    # Calculate the monthly volatility (standard deviation) for SPY and Portfolio
    monthly_volatility_spy = overall_returns['SP500'].std()
    monthly_volatility_portfolio = overall_returns['Portfolio'].std()

    # Annualize the volatility
    annualized_volatility_spy = monthly_volatility_spy * (12 ** 0.5)
    annualized_volatility_portfolio = monthly_volatility_portfolio * (12 ** 0.5)


    # Calculate the average risk-free rate
    avg_rf_rate = port_full_stats['Rf Rate'].mean()

    # Calculate the Sharpe ratio for SPY and Portfolio
    sharpe_ratio_spy = (annualized_return_spy - avg_rf_rate) / annualized_volatility_spy
    sharpe_ratio_portfolio = (annualized_return_portfolio - avg_rf_rate) / annualized_volatility_portfolio


    # Create a DataFrame with all the calculated statistics
    total_stats = pd.DataFrame({
        'Annualized Return': [annualized_return_spy, annualized_return_portfolio],
        'Annualized Volatility': [annualized_volatility_spy, annualized_volatility_portfolio],
        'Sharpe Ratio': [sharpe_ratio_spy, sharpe_ratio_portfolio]
    }, index=['SP500', 'Portfolio'])

    



    # Define independent variable (SPY Excess Return) and dependent variable (Portfolio Excess Return)
    X = month_end_returns['SPY Excess']
    Y = month_end_returns['Portfolio Excess']

    # Add a constant term for the intercept (Jensen's Alpha)
    X = sm.add_constant(X)

    # Run OLS regression
    jensens_model = sm.OLS(Y.astype(float), X.astype(float)).fit()

    # Extract Jensen's Alpha (Intercept)
    jensens_alpha = jensens_model.params['const']



    # Extracting the results from the OLS regression
    alpha = jensens_model.params['const']  # Jensen's Alpha
    beta = jensens_model.params['SPY Excess']  # Beta (SPY Excess coefficient)
    r_squared = jensens_model.rsquared  # R-squared value
    equation = f"Portfolio Excess Return = {alpha:.4f} + ({beta:.4f}) * SPY Excess Return"

    # Create a new DataFrame to store the results
    jensen_results = pd.DataFrame({
        'Alpha (Jensen)': [alpha],
        'Beta': [beta],
        'R-squared': [r_squared],
        'Equation': [equation]
    })


    # Calculate the annualized portfolio return and risk-free rate for the whole period
    portfolio_annualized_return = month_end_returns['Portfolio Returns'].mean() * 12  # Annualized return (monthly average * 12)
    rf_annualized_rate = month_end_returns['Rf Rate'].mean() * 12  # Annualized risk-free rate (monthly average * 12)

    # Now calculate the Treynor Ratio
    treynor_ratio = (portfolio_annualized_return - rf_annualized_rate) / beta

    # Display the Treynor Ratio
    print(f'Treynor Ratio: {treynor_ratio}')

    # Create a new DataFrame to store the Treynor Ratio for the entire period
    treynor_df = pd.DataFrame({
        'Treynor Ratio': [treynor_ratio]
    })


    # Identify the periods where the excess return is negative
    negative_excess_returns = month_end_returns[month_end_returns['Portfolio Excess'] < 0]['Portfolio Excess']

    # Calculate the downside deviation (standard deviation of negative excess returns)
    downside_deviation = negative_excess_returns.std()

    # Now calculate the Sortino Ratio
    sortino_ratio = (portfolio_annualized_return - rf_annualized_rate) / downside_deviation
    # Create a new DataFrame to store the Sortino Ratio for the entire period
    sortino_df = pd.DataFrame({
        'Sortino Ratio': [sortino_ratio]
    })



    # Calculate the tracking error (standard deviation of active returns)
    tracking_error = month_end_returns['Active Return'].std()

    # Calculate the mean active return
    mean_active_return = month_end_returns['Active Return'].mean()

    # Calculate the information ratio
    information_ratio = mean_active_return / tracking_error


    # Create a DataFrame to store the Information Ratio result
    information_ratio_df = pd.DataFrame({
        'Information Ratio': [information_ratio],
        'Tracking Error': [tracking_error],
        'Mean Active Return': [mean_active_return]
    })




    # For Portfolio
    max_drawdown_portfolio = month_end_returns['drawdown_portfolio'].min()
    date_max_drawdown_portfolio = month_end_returns['drawdown_portfolio'].idxmin()

    # For SPY
    max_drawdown_spy = month_end_returns['drawdown_spy'].min()
    date_max_drawdown_spy = month_end_returns['drawdown_spy'].idxmin()

    # Create a new dataframe to store the maximum drawdown details
    drawdown_details = pd.DataFrame({
        'Asset': ['Portfolio', 'SP500'],
        'Max Drawdown': [max_drawdown_portfolio, max_drawdown_spy],
        'Date of Max Drawdown': [date_max_drawdown_portfolio.strftime('%Y-%m-%d'), date_max_drawdown_spy.strftime('%Y-%m-%d')]
    })


    returns_display = month_end_returns[['Portfolio Wealth Curve', 'SPY Wealth Curve', 'Portfolio Returns', 'SPY Returns']].copy()

    # Create tabs
    tab1, tab2 = st.tabs(["📊 Charts", "📋 DataFrames"])

    # 📊 Charts Tab
    with tab1:
        st.subheader("Portfolio Performance Charts")

    fig, ax = plt.subplots(figsize=(10, 6))
    month_end_returns[['Portfolio Wealth Curve', 'SPY Wealth Curve']].plot(ax=ax)

    # Define a custom formatter function for large numbers
    def millions_formatter(x, pos):
        if x >= 1e9:
            return f'{x*1e-9:.1f}B'  # Billions with 1 decimal
        elif x >= 1e6:
            return f'{x*1e-6:.1f}M'  # Millions with 1 decimal
        elif x >= 1e3:
            return f'{x*1e-3:.1f}K'  # Thousands with 1 decimal
        else:
            return f'{x:.0f}'

    # Apply the formatter to y-axis
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))

    ax.set_title("Portfolio vs SPY Wealth Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth")
    st.pyplot(fig)

    # Apply the same formatting to the Prior Peaks plot
    fig, ax = plt.subplots(figsize=(10, 6))
    month_end_returns[['prior_peaks_portfolio', 'prior_peaks_spy']].plot(ax=ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))
    ax.set_title("Prior Peaks for Portfolio and SPY")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    st.pyplot(fig)

    # For the Drawdown plot, keep percentage format
    fig, ax = plt.subplots(figsize=(10, 6))
    month_end_returns[['drawdown_portfolio', 'drawdown_spy']].plot(ax=ax)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Drawdown Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    st.pyplot(fig)

    # 📋 DataFrames Tab
    with tab2:
        st.subheader("Data Tables")

        # Display each DataFrame
        st.write("### Target Portfolio")
        st.dataframe(rebalance_data)

        st.write("### Portfolio vs SP500")
        st.dataframe(returns_display)

        st.write("### Annual Stats")
        st.dataframe(annual_stats)

        st.write("### Total Stats")
        st.dataframe(total_stats)

        st.write("### Jensen's Alpha")
        st.dataframe(jensen_results)

        st.write("### Treynor Ratio")
        st.dataframe(treynor_df)

        st.write("### Sortino Ratio")
        st.dataframe(sortino_df)

        st.write("### Information Ratio")
        st.dataframe(information_ratio_df)

        st.write("### Max Drawdown")
        st.dataframe(drawdown_details)







    # Create an in-memory Excel file
    download_excel = io.BytesIO()

    # Writing the data to the in-memory Excel file

    with pd.ExcelWriter(download_excel, engine='openpyxl', mode='w') as writer:
        # Write the first sheet: Target Portfolio (rebalance_data)
        rebalance_data.to_excel(writer, sheet_name="Target Portfolio", index=False)
        
        # Write the second sheet: Portfolio vs SP500 (combined returns)
        month_end_returns.to_excel(writer, index=False, sheet_name="Portfolio_vs_SP500")
        
        # Write the third sheet: Annual Stats
        annual_stats.to_excel(writer, sheet_name='Annual Stats', index=False)
        
        # Write the fourth sheet: Total Stats
        total_stats.to_excel(writer, sheet_name='Total Stats', index=True)
        
        # Write the fifth sheet: Jensen's Alpha
        jensen_results.to_excel(writer, sheet_name="Jensen's Alpha", index=False)
        
        # Write the sixth sheet: Treynor Ratio
        treynor_df.to_excel(writer, sheet_name='Treynor Ratio', index=False)
        
        # Write the seventh sheet: Sortino Ratio
        sortino_df.to_excel(writer, sheet_name='Sortino Ratio', index=False)
        
        # Write the eighth sheet: Information Ratio
        information_ratio_df.to_excel(writer, sheet_name='Information Ratio', index=False)
        
        # Write the ninth sheet: Max Drawdown
        drawdown_details.to_excel(writer, sheet_name='Max Drawdown', index=False)

    # Ensure writing is finalized
    download_excel.seek(0)

    # Streamlit download button to allow the user to download the Excel file
    st.download_button(
        label="Download Excel file",
        data=download_excel,
        file_name="Portfolio_vs_SP500.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )