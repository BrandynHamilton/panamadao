# Example setup for 'merged_df'
    date_range = pd.date_range(start='2020-01-01', end='2020-12-31', freq='M')
    np.random.seed(0)
    sample_data = np.random.rand(len(date_range)) * 100
    
    # Selecting a single column as a Series for forecasting
    staking_revenue = merged_df['staking revenue']
    
    # Split the data into train and test sets
    train_size = int(len(staking_revenue) * 0.8)
    train, test = staking_revenue.iloc[:train_size], staking_revenue.iloc[train_size:]
    
    # Fit the ARIMA model
    arima_model = ARIMA(train, order=(1,1,1))
    arima_result = arima_model.fit()
    
    # Forecast steps should include the additional future periods you want to forecast
    forecast_steps = len(test) + 1  # Add more steps here for future forecasting
    
    # Generating new index for the forecast that extends beyond the test set
    last_date = train.index[-1] if len(test) == 0 else test.index[-1]
    forecast_index = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='M')[1:]  # This includes one future period
    
    # Forecast
    forecast = arima_result.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()
    
    # Plotting with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Training Data'))
    fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Actual Revenue'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, mode='lines', name='Forecasted Revenue'))
    fig.add_trace(go.Scatter(x=forecast_index, y=confidence_intervals.iloc[:, 0], mode='lines', name='Lower CI', line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=forecast_index, y=confidence_intervals.iloc[:, 1], mode='lines', name='Upper CI', line=dict(color='gray', dash='dot'), fill='tonexty'))
    
    # Show the figure in Streamlit
    st.plotly_chart(fig)

"""
    staking_revenue = merged_df[['staking revenue']].copy()
    staking_revenue['Time'] = np.arange(len(staking_revenue.index))
    
    # Splitting the data
    X = staking_revenue[['Time']]  # Features: Time
    y = staking_revenue['staking revenue']  # Target variable
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Generate time points for the 365-day forecast
    last_time_point = X_train.iloc[-1, 0]
    forecast_time = np.array([[i] for i in range(last_time_point + 1, last_time_point + 366)])  # 365 days out
    
    # Forecast revenue for each day
    forecast_revenue = model.predict(forecast_time)
    
    # Generate dates for the forecast (assuming daily frequency)
    forecast_dates = pd.date_range(start=merged_df.index[-1] + pd.Timedelta(days=1), periods=365, freq='D')
    
    # Create the forecast DataFrame
    forecast_df = pd.DataFrame({'Forecasted Revenue': forecast_revenue}, index=forecast_dates)
    
    # Plotting the data and the forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train['Time'], y=y_train, mode='lines', name='Training Data'))
    fig.add_trace(go.Scatter(x=X_test['Time'], y=y_test, mode='lines', name='Test Data'))
    fig.add_trace(go.Scatter(x=np.arange(last_time_point + 1, last_time_point + 366), y=forecast_revenue, mode='lines', name='Forecasted Revenue', line=dict(color='red')))
    
    # Optionally, you can convert the numeric time points back to dates for the x-axis if needed
    # This step depends on how you want to represent time on the x-axis
    # For simplicity, this example continues to use numeric time points
    
    # Show the figure in Streamlit
    st.plotly_chart(fig)
    
    # Show the forecast DataFrame with the date as the index
    st.write(forecast_df)
        
    # Show the DataFrame
    """