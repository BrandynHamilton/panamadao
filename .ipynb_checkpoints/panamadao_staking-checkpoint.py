#!/usr/bin/env python
# coding: utf-8

# In[281]:


from flipside import Flipside
import time
import pandas as pd
import numpy as np
import requests
import json
import warnings
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')


# In[2]:


balance_sheet_sql = """
WITH RECURSIVE date_series AS (
  SELECT '2023-03-22'::DATE AS day
  UNION ALL
  SELECT day + 1
  FROM date_series
  WHERE day + 1 <= CURRENT_DATE
),
eth_price_daily AS (
    SELECT 
        date_trunc('day', HOUR) AS day, 
        AVG(price) AS avg_price
    FROM ethereum.price.ez_hourly_token_prices
    WHERE symbol = 'WETH'
    GROUP BY date_trunc('day', HOUR)
),
balance_data AS (
    SELECT
        date_trunc('day', block_timestamp) AS day,
        block_timestamp,
        current_bal
    FROM ethereum.core.ez_balance_deltas
    WHERE user_address = lower('0xf1dA938Cbf912b9e5444F6532C20A58d09Dd67B8')
),
filled_balances AS (
    SELECT
        ds.day AS day,
        COALESCE(
            bd.current_bal, 
            LAST_VALUE(bd.current_bal IGNORE NULLS) OVER (
                ORDER BY ds.day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
        ) AS filled_balance,
        ROW_NUMBER() OVER (PARTITION BY ds.day ORDER BY bd.block_timestamp DESC) AS rn
    FROM date_series ds
    LEFT JOIN balance_data bd ON ds.day = bd.day
),
assets AS (
    SELECT 
        fb.day, 
        fb.filled_balance * ep.avg_price AS balance_in_usd
    FROM filled_balances fb
    LEFT JOIN eth_price_daily ep ON fb.day = ep.day
    WHERE fb.rn = 1
),
equity_sales AS (
    SELECT
        date_trunc('day', block_timestamp) AS day,
        ethereum.public.udf_hex_to_int(SUBSTR(DATA, 1, 66)) AS ID,
        ethereum.public.udf_hex_to_int(substr(data, 131, 64)) / power(10, 18) AS amount_eth
    FROM ethereum.core.fact_event_logs
    WHERE contract_address = lower('0x93519f3558775BBd5c0d501A2fD7a58bb034B379') AND
              topics[0] = lower('0xc9f72b276a388619c6d185d146697036241880c36654b1a3ffdad07c24038d99')
),
daily_data AS (
    SELECT
        ds.day,
        COALESCE(es.transaction_value_eth, 0) AS transaction_value_eth,
        COALESCE(es.unique_units, 0) AS unique_units
    FROM date_series ds
    LEFT JOIN (
        SELECT 
            day,
            SUM(amount_eth) AS transaction_value_eth,
            COUNT(DISTINCT ID) AS unique_units
        FROM equity_sales
        GROUP BY day
    ) es ON ds.day = es.day
),
rolling_avg_price AS (
    SELECT
        day,
        AVG(transaction_value_eth / NULLIF(unique_units, 0)) OVER (ORDER BY day ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS avg_price_last_30_days
    FROM daily_data
),
cumulative_units AS (
    SELECT
        day,
        SUM(unique_units) OVER (ORDER BY day) AS cumulative_units
    FROM daily_data
),
market_cap AS (
    SELECT
        rap.day,
        COALESCE(
            rap.avg_price_last_30_days * ep.avg_price,
            LAST_VALUE(rap.avg_price_last_30_days * ep.avg_price IGNORE NULLS) OVER (
                ORDER BY rap.day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
        ) AS avg_price_last_30_days_usd,
        cu.cumulative_units,
        -1 * COALESCE(
            rap.avg_price_last_30_days * cu.cumulative_units * ep.avg_price,
            LAST_VALUE(rap.avg_price_last_30_days * cu.cumulative_units * ep.avg_price IGNORE NULLS) OVER (
                ORDER BY rap.day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
        ) AS market_capitalization_usd
    FROM rolling_avg_price rap
    LEFT JOIN eth_price_daily ep ON rap.day = ep.day
    JOIN cumulative_units cu ON rap.day = cu.day
)

SELECT 
    a.day, 
    a.balance_in_usd as assets, 
    mc.avg_price_last_30_days_usd, 
    mc.cumulative_units, 
    mc.market_capitalization_usd as equity,
    -mc.market_capitalization_usd as market_cap 
FROM assets a
JOIN market_cap mc ON a.day = mc.day
ORDER BY a.day desc;


"""


# In[3]:


income_statement_sql = """

WITH auctions AS (
    SELECT
        date_trunc('Month', BLOCK_TIMESTAMP) AS month,
        SUM(ethereum.public.udf_hex_to_int(substr(data, 131, 64)) / power(10, 18)) AS auction_revenue_eth
    FROM ethereum.core.fact_event_logs
    WHERE contract_address = lower('0x93519f3558775BBd5c0d501A2fD7a58bb034B379') AND
          topics[0] = lower('0xc9f72b276a388619c6d185d146697036241880c36654b1a3ffdad07c24038d99')
    GROUP BY date_trunc('Month', BLOCK_TIMESTAMP)
),
seed_funding AS (
    SELECT 
        date_trunc('Month', BLOCK_TIMESTAMP) AS month, 
        SUM(ETH_VALUE) AS eth_value
    FROM ethereum.core.fact_traces
    WHERE TO_ADDRESS = lower('0xf1dA938Cbf912b9e5444F6532C20A58d09Dd67B8')
    GROUP BY date_trunc('Month', BLOCK_TIMESTAMP)
),
eth_price AS (
    SELECT 
        date_trunc('Month', HOUR) AS month, 
        avg(price) AS avg_price
    FROM ethereum.price.ez_hourly_token_prices
    WHERE symbol = 'WETH'
    GROUP BY date_trunc('Month', HOUR)
),
combined_revenue AS (
    SELECT
        sf.month,
        sf.eth_value AS total_revenue_eth,
        a.auction_revenue_eth,
        (sf.eth_value - COALESCE(a.auction_revenue_eth, 0)) AS non_auction_revenue_eth,
        sf.eth_value * ep.avg_price AS total_revenue_usd,
        COALESCE(a.auction_revenue_eth, 0) * ep.avg_price AS auction_revenue_usd,
        (sf.eth_value - COALESCE(a.auction_revenue_eth, 0)) * ep.avg_price AS non_auction_revenue_usd
    FROM seed_funding sf
    LEFT JOIN auctions a ON sf.month = a.month
    LEFT JOIN eth_price ep ON sf.month = ep.month
),
expenses AS (
    SELECT 
        date_trunc('Month', BLOCK_TIMESTAMP) AS month, 
        -SUM(ethereum.public.udf_hex_to_int(substr(data, 451, 64)) / power(10, 18)) AS expenses_eth
    FROM ethereum.core.fact_event_logs
    WHERE contract_address = lower('0xf1dA938Cbf912b9e5444F6532C20A58d09Dd67B8') AND
          topics[0] = lower('0x7e74d8579043af873f575ed17043a48d6beba2668c6b53325bcd8c9a550e5e9c')
    GROUP BY date_trunc('Month', BLOCK_TIMESTAMP)
),
monthly_expenses AS (
    SELECT 
        e.month,
        e.expenses_eth,
        e.expenses_eth * ep.avg_price AS expenses_usd
    FROM expenses e
    LEFT JOIN eth_price ep ON e.month = ep.month
)
SELECT
    cr.month,
    cr.total_revenue_eth,
    cr.auction_revenue_eth,
    cr.non_auction_revenue_eth,
    cr.total_revenue_usd,
    cr.auction_revenue_usd,
    cr.non_auction_revenue_usd,
    COALESCE(me.expenses_eth, 0) AS expenses_eth,
    COALESCE(me.expenses_usd, 0) AS expenses_usd,
    COALESCE(cr.total_revenue_eth, 0) + COALESCE(me.expenses_eth, 0) AS net_income_eth,
    COALESCE(cr.total_revenue_usd, 0) + COALESCE(me.expenses_usd, 0) AS net_income_usd
FROM combined_revenue cr
FULL OUTER JOIN monthly_expenses me ON cr.month = me.month
ORDER BY cr.month DESC;


"""


# In[4]:


api_key = '4604d4d2-ccf3-4864-90b4-db6bf13c663b'
api_key_cg = 'CG-jTsiV2rsyVSHHULoNSWHU493'


# In[5]:

@st.cache_data()
def createQueryRun(sql):
    url = "https://api-v2.flipsidecrypto.xyz/json-rpc"

    payload = json.dumps({
      "jsonrpc": "2.0",
      "method": "createQueryRun",
      "params": [
        {
          "resultTTLHours": 1,
          "maxAgeMinutes": 0,
          "sql": sql,
          "tags": {
            "source": "postman-demo",
            "env": "test"
          },
          "dataSource": "snowflake-default",
          "dataProvider": "flipside"
        }
      ],
      "id": 1
    })
    headers = {
      'Content-Type': 'application/json',
      'x-api-key': api_key
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = json.loads(response.text)
    query_run_id = response_data['result']['queryRun']['id']
    #total_pages = response_data['result']['queryRun']['totalSize']
    
    return response_data, query_run_id#, total_pages, 
     

    if 'error' in response_data:
        print("Error:", response_data['error']['msg'])
    else:
        print("Query Run ID:", query_run_id)


# In[6]:

@st.cache_data()
def getQueryResults(query_run_id, retry_delay=5, max_retries=12):
    """
    Fetches query results, with retries to handle asynchronous completion of queries.
    
    :param query_run_id: ID of the query run to fetch results for.
    :param retry_delay: Time in seconds to wait between retries.
    :param max_retries: Maximum number of retries before giving up.
    :return: JSON response containing the query results.
    """
    url = "https://api-v2.flipsidecrypto.xyz/json-rpc"
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "getQueryRunResults",
        "params": [{"queryRunId": query_run_id, "format": "json", "page": {"number": 1, "size": 100}}],
        "id": 1
    })
    headers = {'Content-Type': 'application/json', 'x-api-key': api_key}
    
    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, data=payload)
        resp_json = response.json()
        
        if 'result' in resp_json and 'rows' in resp_json['result']:  # Check if results are ready
            return resp_json
        elif 'error' in resp_json and resp_json['error'].get('message') == 'QueryRunNotFinished':
            print(f"Query not finished, waiting {retry_delay} seconds before retry {attempt + 1}/{max_retries}.")
            time.sleep(retry_delay)
        else:
            print(f"Failed to fetch results: {resp_json.get('error', {}).get('message', 'Unknown error')}")
            break
    else:  # If we've exhausted retries
        print("Query did not finish in time or failed to fetch results after multiple retries.")
        return None

# Example usage
bs_response_data, q_id = createQueryRun(balance_sheet_sql)


# In[8]:


print(bs_response_data['result']['queryRun'])


# In[12]:


#bs_resp_d, bs_query_run_id = createQueryRun(balance_sheet_sql)
bs_df = getQueryResults(q_id)


# In[163]:


bs_df


# In[164]:


bs_dataframe = pd.DataFrame(bs_df['result']['rows'])


# In[165]:


bs_dataframe['day'] = pd.to_datetime(bs_dataframe['day'])
bs_dataframe.set_index('day', inplace=True)
bs_dataframe


# In[16]:


is_response, is_query_run_id= createQueryRun(income_statement_sql)


# In[19]:


is_df = getQueryResults(is_query_run_id)


# In[20]:


is_dataframe = pd.DataFrame(is_df['result']['rows'])
is_dataframe['month'] = pd.to_datetime(is_dataframe['month'])
is_dataframe.set_index('month', inplace=True)


# In[21]:


is_dataframe


# In[155]:


bs_dataframe['assets'].describe()


# In[166]:


# Replace values greater than $50,000 with $50,000
bs_dataframe.loc[bs_dataframe['assets'] > 50000, 'assets'] = 50000


# In[168]:


bs_dataframe.plot()


# In[169]:


is_dataframe.plot()


# In[170]:

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


# In[171]:


api_key_dune = 'w3Tusb5XCOyAMb2ggKHkKuVERuXHnwyW'


# In[172]:


lidostaking_url = "https://api.dune.com/api/v1/query/570874/results/"
staking_params = {"api_key": api_key_dune}
staking_timeseries = fetch_data_from_api(lidostaking_url, staking_params)


# In[173]:


staking_timeseries['time'] = pd.to_datetime(staking_timeseries['time'])
staking_timeseries.set_index('time', inplace=True)


# In[174]:


staking_timeseries


# In[175]:


staking = staking_timeseries['Lido staking APR(instant)'].to_frame('APR')


# In[176]:


bs_dataframe


# In[177]:


staking


# In[178]:


bs_dataframe.index = bs_dataframe.index.normalize()


# In[179]:


staking.index = staking.index.normalize()


# In[180]:


merged_df = pd.merge(bs_dataframe, staking, left_index=True, right_index=True, how='inner')


# In[181]:


merged_df = merged_df.iloc[::-1]


# In[182]:


merged_df['APR_daily'] = merged_df['APR'] / 100 / 365
merged_df['APR_Decimal'] = merged_df['APR'] / 100


# In[183]:


# Parameters
x_percent = 50 / 100  # 50%


# In[282]:


merged_df['staked_ETH'] = merged_df['assets'] * x_percent


# In[283]:


merged_df['simple_interest_daily'] = merged_df['staked_ETH'] * merged_df['APR_daily']
merged_df['simple_interest_accumulated'] = merged_df['simple_interest_daily'].cumsum()


# In[284]:


merged_df


# In[187]:


# Assuming merged_df is your DataFrame

# Initialize compound interest for the first day with staked ETH
merged_df['compound_interest'] = merged_df['staked_ETH'].iloc[0]

# Loop through each day to calculate compound interest
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


# In[188]:


merged_df['compound_interest_daily'] = merged_df['total_compound_interest'].diff().fillna(0)


# In[189]:


merged_df


# In[190]:


merged_df[['simple_interest_accumulated', 'total_compound_interest']].plot()


# In[191]:


merged_df


# In[192]:


cumulative_compound_interest = merged_df['total_compound_interest'].iloc[-1]
cumulative_simple_interest = merged_df['simple_interest_accumulated'].iloc[-1]


# In[193]:


cumulative_simple_interest


# In[194]:


cumulative_compound_interest


# In[195]:


sim_balance_sheet = merged_df.copy()
sim_balance_sheet['simulated_assets_simple_interest'] = sim_balance_sheet['assets'] + sim_balance_sheet['simple_interest_daily']
sim_balance_sheet['simulated_assets_compound_interest'] = sim_balance_sheet['assets'] + sim_balance_sheet['compound_interest_daily']


# In[196]:


sim_balance_sheet[['assets','simulated_assets_simple_interest','simulated_assets_compound_interest']]


# In[197]:


monthly_yield = merged_df[['simple_interest_daily','compound_interest_daily']].resample('M').sum()

monthly_yield.index = monthly_yield.index.normalize()


# In[198]:


monthly_yield


# In[199]:


is_dataframe.index = is_dataframe.index.normalize()


# In[200]:


is_dataframe_monthly = is_dataframe.resample('M').sum()
is_dataframe_monthly


# In[201]:


income_stmt_sim = is_dataframe_monthly[['total_revenue_usd','auction_revenue_usd','expenses_usd','net_income_usd']].merge(monthly_yield, left_index=True, right_index=True)


# In[202]:


income_stmt_sim['simulated_rev_simple_interest'] = income_stmt_sim['total_revenue_usd'] + income_stmt_sim['simple_interest_daily']
income_stmt_sim['simulated_rev_compound_interest'] = income_stmt_sim['total_revenue_usd'] + income_stmt_sim['compound_interest_daily']

income_stmt_sim['simple_interest_net_income'] = income_stmt_sim['simulated_rev_simple_interest'] + income_stmt_sim['expenses_usd']
income_stmt_sim['compound_interest_net_income'] = income_stmt_sim['simulated_rev_compound_interest'] + income_stmt_sim['expenses_usd']


# In[203]:


income_stmt_sim['simple_interest_income_diff'] = income_stmt_sim['simple_interest_net_income'] - income_stmt_sim['net_income_usd']
income_stmt_sim['compound_interest_income_diff'] = income_stmt_sim['compound_interest_net_income'] - income_stmt_sim['net_income_usd']


# In[204]:


income_stmt_sim


# In[205]:


income_stmt_sim['net_income_usd_nozero'] = income_stmt_sim['net_income_usd'].replace({0: 1})


income_stmt_sim


# In[206]:


income_stmt_sim['simple_interest_percent_chg'] = income_stmt_sim['simple_interest_income_diff'] / abs(income_stmt_sim['net_income_usd_nozero'])
income_stmt_sim['compound_interest_percent_chg'] = income_stmt_sim['compound_interest_income_diff'] / abs(income_stmt_sim['net_income_usd_nozero'])


# In[207]:


income_stmt_sim[['net_income_usd','simple_interest_net_income','compound_interest_net_income']]


# In[208]:


income_stmt_sim[['simple_interest_percent_chg','compound_interest_percent_chg']]


# In[209]:


income_stmt_sim['equity_investments_usd'] = income_stmt_sim['total_revenue_usd'] - income_stmt_sim['auction_revenue_usd']


# In[285]:


income_stmt_sim = income_stmt_sim.drop(columns=['simple_interest_income_diff'])


# In[211]:


income_stmt_sim = income_stmt_sim.drop(columns=['compound_interest_income_diff'])


# In[286]:


income_stmt_sim


# In[213]:


revenue_sources = income_stmt_sim[['total_revenue_usd','auction_revenue_usd','simple_interest_daily','compound_interest_daily']]


# In[214]:


revenue_sources['equity_investments_usd'] = revenue_sources['total_revenue_usd'] - revenue_sources['auction_revenue_usd']


# In[216]:


revenue_sources = revenue_sources.drop(columns=['total_revenue_usd'])


# In[217]:


cumulative_revenue_sources = revenue_sources.sum()


# In[218]:


cumulative_revenue_sources


# In[219]:


compound_interest_pie = cumulative_revenue_sources[['auction_revenue_usd','compound_interest_daily','equity_investments_usd']].plot.pie()


# In[220]:


simple_interest_pie = cumulative_revenue_sources[['auction_revenue_usd','simple_interest_daily','equity_investments_usd']].plot.pie()


# In[221]:


revenue_sources['auction_revenue_usd'].describe()


# In[222]:


revenue_sources['compound_interest_daily'].describe()


# In[224]:


# Assuming 'revenue_sources' is your DataFrame
interest_more_than_auctions = revenue_sources[revenue_sources['simple_interest_daily'] > revenue_sources['auction_revenue_usd']]
 


# In[280]:


income_stmt_sim[['simple_interest_daily','auction_revenue_usd']].plot()


# In[246]:


interest_vs_auctions = px.bar(
    data_frame = income_stmt_sim,
    x = income_stmt_sim.index,
    y = ['auction_revenue_usd', 'simple_interest_daily'],
    barmode = 'relative'
    
)


# In[247]:


interest_vs_auctions


# In[264]:


net_income_comparison = income_stmt_sim[['net_income_usd','simple_interest_net_income','simple_interest_percent_chg']]


# In[275]:


income_pct_chg = px.bar(
    data_frame = net_income_comparison,
    x = income_stmt_sim.index,
    y = ['simple_interest_percent_chg']
   
    
)


# In[276]:


income_pct_chg


# In[277]:


#Shows how much net income increases from adding staking yield; in months with 0 income, the staking yield results in high % increase
#75% of time, staking yield boosts income by 43% or lower

net_income_comparison['simple_interest_percent_chg'].describe()


# In[287]:


#Can expect average of $50 per month from staking
#25th percentile for interest is higher than auction revenue 25th percentile
income_stmt_sim['simple_interest_daily'].describe()


# In[290]:


#Half of auction revenues per month are under $144
income_stmt_sim['auction_revenue_usd'].describe()


# In[291]:


revenue_sources['auction_revenue_usd'].plot.hist()
#monthly_revenue from auctions usually under $500


# In[292]:


#3 months where staking revenue was larger than auction revenue

interest_more_than_auctions


# In[294]:


income_stmt_sim['simple_interest_percent_chg'].describe()


# In[ ]:




