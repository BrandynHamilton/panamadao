from flipside import Flipside
import pandas as pd
import numpy as np
import requests
import json
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import matplotlib.pyplot as plt
import time
from scripts import balance_sheet_sql, income_statement_sql
#from panamadao_staking import balance_sheet_sql, income_statement_sql
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import streamlit as st
from pandas.tseries.offsets import MonthEnd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

api_key = "4604d4d2-ccf3-4864-90b4-db6bf13c663b"
api_key_dune = 'w3Tusb5XCOyAMb2ggKHkKuVERuXHnwyW'

@st.cache_data()
def createQueryRun(sql):
    url = "https://api-v2.flipsidecrypto.xyz/json-rpc"
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "createQueryRun",
        "params": [{
            "resultTTLHours": 1,
            "maxAgeMinutes": 0,
            "sql": sql,
            "tags": {"source": "streamlit-demo", "env": "test"},
            "dataSource": "snowflake-default",
            "dataProvider": "flipside"
        }],
        "id": 1
    })
    headers = {'Content-Type': 'application/json', 'x-api-key': api_key}
    response = requests.post(url, headers=headers, data=payload)
    response_data = response.json()
    if 'error' in response_data:
        st.error("Error: " + response_data['error']['message'])
        return None, None
    query_run_id = response_data['result']['queryRun']['id']
    return response_data, query_run_id

@st.cache_data()
def getQueryResults(query_run_id, attempts=10, delay=30):
    """Fetch query results with retries for asynchronous completion."""
    url = "https://api-v2.flipsidecrypto.xyz/json-rpc"
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "getQueryRunResults",
        "params": [{"queryRunId": query_run_id, "format": "json", "page": {"number": 1, "size": 10000}}],
        "id": 1
    })
    headers = {'Content-Type': 'application/json', 'x-api-key': api_key}

    for attempt in range(attempts):
        response = requests.post(url, headers=headers, data=payload)
        resp_json = response.json()
        if 'result' in resp_json:
            return resp_json  # Data is ready
        elif 'error' in resp_json and 'message' in resp_json['error'] and 'not yet completed' in resp_json['error']['message']:
            time.sleep(delay)  # Wait for a bit before retrying
        else:
            break  # Break on unexpected error
    return None  # Return None if data isn't ready after all attempts


@st.cache_data()
def fetch_data_from_api(api_url, params=None):
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'rows' in data['result']:
            return pd.DataFrame(data['result']['rows'])
        return data
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return pd.DataFrame()  # or an empty dict

#Example SQL query, replace with your actual query
    
bs_response_data, q_id_bs = createQueryRun(balance_sheet_sql)
if q_id_bs:
    bs_df_json = getQueryResults(q_id_bs)
    if bs_df_json:
        # Process and display the balance sheet data
        bs_df = pd.DataFrame(bs_df_json['result']['rows'])
        #st.write(bs_df)

# Automatically fetch income statement data on app load
is_response_data, q_id_is = createQueryRun(income_statement_sql)
if q_id_is:
    is_df_json = getQueryResults(q_id_is)
    if is_df_json:
        # Process and display the income statement data
        is_df = pd.DataFrame(is_df_json['result']['rows'])
        #st.write(is_df)

bs_dataframe = bs_df 
is_dataframe = is_df 

bs_dataframe['day'] = pd.to_datetime(bs_dataframe['day'])
bs_dataframe.set_index('day', inplace=True)

is_dataframe['month'] = pd.to_datetime(is_dataframe['month'])
is_dataframe.set_index('month', inplace=True)

lidostaking_url = "https://api.dune.com/api/v1/query/570874/results/"
staking_params = {"api_key": api_key_dune}
staking_timeseries = fetch_data_from_api(lidostaking_url, staking_params)
staking_timeseries['time'] = pd.to_datetime(staking_timeseries['time'])
staking_timeseries.set_index('time', inplace=True)
staking = staking_timeseries['Lido staking APR(instant)'].to_frame('APR')

bs_dataframe.index = bs_dataframe.index.normalize()
staking.index = staking.index.normalize()

merged_df = pd.merge(bs_dataframe, staking, left_index=True, right_index=True, how='inner')
merged_df = merged_df.iloc[::-1]
merged_df['APR_daily'] = merged_df['APR'] / 100 / 365
merged_df['APR_Decimal'] = merged_df['APR'] / 100
    

def app_page():

    panama_dao_logo = "images/diablicos_logo.png"

    st1, st2 = st.columns(2)
    
    st.image(panama_dao_logo, width=50)
    
    st.title('Panama Dao Staking Analysis')

    
    x_percent = st.slider('Select percentage of assets to stake:', min_value=0, max_value=100, value=50) / 100.0

    merged_df['staked_ETH'] = merged_df['assets'] * x_percent

    merged_df['simple_interest_daily'] = merged_df['staked_ETH'] * merged_df['APR_daily']
    merged_df['simple_interest_accumulated'] = merged_df['simple_interest_daily'].cumsum()
    merged_df['compound_interest'] = merged_df['staked_ETH'].iloc[0]
    
    for i in range(1, len(merged_df)):
        # Calculate the increase in staked_ETH from the previous day, if any
        daily_increase = merged_df['staked_ETH'].iloc[i] - merged_df['staked_ETH'].iloc[i-1]
        
        # Calculate new principal for the day
        new_principal = merged_df['compound_interest'].iloc[i-1] + daily_increase
        
        # Calculate today's compound interest
        daily_rate = merged_df['APR_daily'].iloc[i]
        merged_df.at[merged_df.index[i], 'compound_interest'] = new_principal * (1 + daily_rate)
    
    # Calculate total compound interest by subtracting the initial staked ETH from the compound interest for each day
    merged_df['total_compound_interest'] = merged_df['compound_interest'] - merged_df['staked_ETH']
    merged_df['absolute_compound_return'] = merged_df['compound_interest'] - merged_df['staked_ETH']

# Calculate the return on compound interest as a percentage of the original staked_ETH
    merged_df['compound_return_percentage'] = (merged_df['absolute_compound_return'] / merged_df['staked_ETH'])
    merged_df['staking revenue'] = merged_df['total_compound_interest'].diff().fillna(0)
    
    cumulative_compound_interest = merged_df['staking revenue'].sum()
    cumulative_simple_interest = merged_df['simple_interest_accumulated'].iloc[-1]
    
    sim_balance_sheet = merged_df.copy()
    sim_balance_sheet['simulated_assets_simple_interest'] = sim_balance_sheet['assets'] + sim_balance_sheet['simple_interest_daily']
    sim_balance_sheet['simulated_assets_compound_interest'] = sim_balance_sheet['assets'] + sim_balance_sheet['staking revenue']
    
    monthly_yield = merged_df[['simple_interest_daily','staking revenue']].resample('M').sum()
    monthly_yield.index = monthly_yield.index.normalize()
    
    is_dataframe.index = is_dataframe.index.normalize()
    is_dataframe_monthly = is_dataframe.resample('M').sum()
    
    income_stmt_sim = is_dataframe_monthly[['total_revenue_usd','auction_revenue_usd','expenses_usd','net_income_usd']].merge(monthly_yield, left_index=True, right_index=True)
    
    income_stmt_sim['simulated_rev_simple_interest'] = income_stmt_sim['total_revenue_usd'] + income_stmt_sim['simple_interest_daily']
    income_stmt_sim['simulated_rev_compound_interest'] = income_stmt_sim['total_revenue_usd'] + income_stmt_sim['staking revenue']
    
    income_stmt_sim['simple_interest_net_income'] = income_stmt_sim['simulated_rev_simple_interest'] + income_stmt_sim['expenses_usd']
    income_stmt_sim['compound_interest_net_income'] = income_stmt_sim['simulated_rev_compound_interest'] + income_stmt_sim['expenses_usd']

    income_stmt_sim['net_income_usd_nozero'] = income_stmt_sim['net_income_usd'].replace({0: 1})
    
    income_stmt_sim['simple_interest_income_diff'] = income_stmt_sim['simple_interest_net_income'] - income_stmt_sim['net_income_usd']
    income_stmt_sim['compound_interest_income_diff'] = income_stmt_sim['compound_interest_net_income'] - income_stmt_sim['net_income_usd']
    
    
    income_stmt_sim['simple_interest_percent_chg'] = income_stmt_sim['simple_interest_income_diff'] / abs(income_stmt_sim['net_income_usd_nozero'])
    income_stmt_sim['compound_interest_percent_chg'] = income_stmt_sim['compound_interest_income_diff'] / abs(income_stmt_sim['net_income_usd_nozero'])
    
    income_stmt_sim['equity_investments_usd'] = income_stmt_sim['total_revenue_usd'] - income_stmt_sim['auction_revenue_usd']
    income_stmt_sim = income_stmt_sim.drop(columns=['simple_interest_income_diff'])
    #income_stmt_sim = income_stmt_sim.drop(columns=['compound_interest_income_diff'])
    income_stmt_sim['Difference']= income_stmt_sim['compound_interest_net_income'] - income_stmt_sim['net_income_usd']
    income_stmt_sim.index = pd.to_datetime(income_stmt_sim.index)
    #income_stmt_sim.index = income_stmt_sim.index.date
    
    revenue_sources = income_stmt_sim[['total_revenue_usd','auction_revenue_usd','staking revenue']]
    revenue_sources['Other Income (Equity)'] = revenue_sources['total_revenue_usd'] - revenue_sources['auction_revenue_usd']
    revenue_sources = revenue_sources.drop(columns=['total_revenue_usd'])
    
    cumulative_revenue_sources = revenue_sources.sum()
    compound_interest_pie = cumulative_revenue_sources[['auction_revenue_usd','staking revenue']].plot.pie()
    
    
    interest_more_than_auctions = revenue_sources[revenue_sources['staking revenue'] > revenue_sources['auction_revenue_usd']]
    interest_more_than_auctions = interest_more_than_auctions.drop(columns=['Other Income (Equity)'])
    interest_more_than_auctions.index = interest_more_than_auctions.index.date
    
    formatted_interest_more_than_auctions = interest_more_than_auctions.iloc[::-1].style.format("${0:,.2f}")
    
    interest_vs_auctions = px.bar(
        data_frame = income_stmt_sim,
        x = income_stmt_sim.index,
        y = ['auction_revenue_usd', 'staking revenue'],
        barmode = 'relative'
        
    )

    
    
    net_income_comparison = px.bar(
        data_frame = income_stmt_sim,
        x = income_stmt_sim.index,
        y = ['net_income_usd','compound_interest_net_income'],
        barmode = 'group'

        
    )
    
    

    
    fig, ax = plt.subplots()
    revenue_sources['auction_revenue_usd'].plot.hist(bins=20, alpha=0.5, ax=ax)
    plt.xlabel('Auction Revenue USD')  # Optional: Add an X-axis label
    plt.ylabel('Frequency')  # Optional: Add a Y-axis label
    #plt.title('Histogram of Auction Revenue USD')  # Optional: Add a title


    data = cumulative_revenue_sources[['auction_revenue_usd', 'staking revenue','Other Income (Equity)']]
    values = data.values.flatten()  # Ensure the values are in the correct shape (flatten if necessary)
    fig1 = go.Figure(data=[go.Pie(labels=data.index, values=data.values.flatten())])
    fig1.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=20)
    #fig1.update_layout(title='Cumulative Revenue Sources')


    # Display Plotly figure in Streamlit

    

    
    
    
    
    st.subheader('Summary')
    st.write("""
    Our analysis focused on the potential financial benefits of staking 50% of PanamaDAO's ETH treasury.
    Staking is an investment strategy where cryptocurrency is locked to receive rewards, similar to earning interest in a savings account.

    We examined the change in net income due to compounded staking returns, termed compound_interest_percent_chg.
    Staking has the potential to enhance net income, providing additional revenue beyond regular operations.

    The compound_interest_percent_chg ranged from a modest 0.19% to a significant 128%, indicating varying monthly income enhancements due to staking.
    In several months, staking revenue surpassed auction revenue, highlighting its potential as a substantial income source.
    On average, staking added a 4.48% return to the net income, contributing to an additional $658.53 over the observed period.
    
    """)
    st.subheader('Reccomendations')
    st.write(""" - Consider staking as part of the treasury management strategy to diversify and stabilize income streams.""")
    
    st.write("""- Use the slider at the top to adjust the staked percentage based on the DAO's operational needs and market conditions to optimize returns.""")           
    
    
    
    

    st.subheader('Revenue Analysis')

    

    formatted_cumulative_rev = pd.DataFrame({
        'Revenue Source': ['Auction Revenue','Staking Yield','Other Income (Equity)'],
        'Amount': [f"${cumulative_revenue_sources['auction_revenue_usd']:,.2f}", f"${cumulative_revenue_sources['staking revenue']:,.2f}", f"${cumulative_revenue_sources['Other Income (Equity)']:,.2f}" ]
        
    })




    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Cumulative Compound Interest',f"${cumulative_compound_interest:,.2f}")

    with col2:
        st.metric('Return on Staking',f"{merged_df['compound_return_percentage'].iloc[-1]:.2%}")
    with col3:
        st.write('Cumulative Revenue by Source to Date')
        st.table(formatted_cumulative_rev.set_index('Revenue Source'))
        

    

    st.write('Operational Revenue Composition')
    st.plotly_chart(fig1, use_container_width=True)

    income_stmt_sim.index = income_stmt_sim.index.date
    formatted_income_stmt = income_stmt_sim[['staking revenue', 'auction_revenue_usd']].iloc[::-1].style.format("${0:,.2f}")


    
    


        
    st.subheader('Auction vs Staking Revenue')
    st.write('The bar chart presents a comparative view of auction revenue versus staking revenue over time.')
    st.plotly_chart(interest_vs_auctions, use_container_width=True)

    st.write('Monthly staking & auction revenue over time')
    st.dataframe(formatted_income_stmt, use_container_width=True)

    st.write('Highlights specific months where staking revenue was greater than auction revenue, showcasing the impact of staking during periods of lower auction sales.')
    st.table(formatted_interest_more_than_auctions)
    
    st.write('For comparison, histogram Shows Monthly Revenue from Auctions Usually Under $500')
    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    
        
The auction revenue summary provides insights into the variability and spread of monthly earnings. The mean and standard deviation highlight overall performance and fluctuation, respectively. The median and quartile values offer a deeper understanding of the distribution, with the median showing the middle point of revenue, and quartiles revealing the range within which the majority of revenues fall, indicating periods of both lower and higher earnings.




""")
    comparison_col1, comparison_col2 = st.columns(2)

    with comparison_col1:
    
        
        st.table(income_stmt_sim['auction_revenue_usd'].describe())

    with comparison_col2:
        
        st.table(income_stmt_sim['staking revenue'].describe())
        
    
    
    
    st.subheader('Net Income Analysis')

         # Correct approach: Calculate the median excluding the outlier
    # Use != to exclude the outlier from the median calculation
    outlier = income_stmt_sim['compound_interest_percent_chg'].max()
    median_without_outlier = income_stmt_sim['compound_interest_percent_chg'][income_stmt_sim['compound_interest_percent_chg'] != outlier].mean()

    # Replace the outlier value with the calculated median
    income_stmt_sim.loc[income_stmt_sim['compound_interest_percent_chg'] >= outlier, 'compound_interest_percent_chg'] = median_without_outlier
    
    # Verify the replacement





    

    formatted_income_stmt_2 = income_stmt_sim[['net_income_usd', 'compound_interest_net_income']].iloc[::-1].style.format("${0:,.2f}")

    formatted_income_stmt_3 = income_stmt_sim[['compound_interest_percent_chg']].iloc[::-1].style.format("${0:.2%}")


    st.write('''

    The comparison of actual and simulated net incomes illustrates the potential impact of compounded staking interest on the DAO's financial performance. By analyzing the shifts from actual to simulated figures, stakeholders can gauge how staking contributions might mitigate losses or amplify profits across different periods. This analysis underscores the strategic value of staking in enhancing the DAO's overall financial health, suggesting a nuanced approach to managing and optimizing staking practices for better fiscal outcomes.









''')
  


    #st.plotly_chart(net_income_comparison)
    st.dataframe(formatted_income_stmt_2, use_container_width=True)

    

        # Calculate the median excluding values greater than 101%
    
    
    # Replace values greater than 101% with the calculated median
    income_stmt_sim.loc[income_stmt_sim['compound_interest_percent_chg'] > outlier, 'compound_interest_percent_chg'] = median_without_outlier

    
    # Now you can create your Plotly figure as before
    fig2 = px.line(
        income_stmt_sim,
        x=income_stmt_sim.index,  # Replace 'index' with the name of your datetime column if it's different
        y=['compound_interest_percent_chg']
    )
    
    fig2.update_layout(
        yaxis=dict(
            tickformat=".0%",  # No decimal places for percentage
            title="Compound Interest Percent Change"
        )
    )

    # Display the updated figure in the Streamlit app

    compound_formatted = income_stmt_sim[['compound_interest_percent_chg']].iloc[::-1].style.format("{0:.2%}")
    st.write("""
    The compound_interest_percent_chg metric in our income statements quantifies the impact of compounded staking returns on our DAO's net income, highlighting how staking enhances financial performance beyond standard operations. For example, a 33% increase in August 2023 signifies substantial growth in net income solely from staking. This data underscores staking's contribution to our financial expansion, illustrating its potential as a robust income stream.





    """)
    st.plotly_chart(fig2, use_container_width=True)

    
    col4, col5 = st.columns(2)
    
    
    

        
    
    
    
        
    sim_balance_sheet.index = sim_balance_sheet.index.date
    formatted_balance_sheet = sim_balance_sheet[['assets','simulated_assets_compound_interest']].iloc[::-1].style.format("${0:,.2f}")
    
        


    st.subheader('Balance Sheet Analysis')
    st.write('Actual vs Simulated Assets')
    st.dataframe(formatted_balance_sheet, use_container_width=True)


    
    

    
    
    
   
    



    

    

app_page()
    
    
    
    
    
    
    
    
    
    
