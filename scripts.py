balance_sheet_sql = """
WITH RECURSIVE date_series AS (
  SELECT
    '2023-03-22' :: DATE AS day
  UNION
  ALL
  SELECT
    day + 1
  FROM
    date_series
  WHERE
    day + 1 <= CURRENT_DATE
),
eth_price_daily AS (
  SELECT
    date_trunc('day', HOUR) AS day,
    AVG(price) AS avg_price
  FROM
    ethereum.price.ez_prices_hourly
  WHERE
    symbol = 'WETH'
  GROUP BY
    date_trunc('day', HOUR)
),
balance_data AS (
  SELECT
    date_trunc('day', block_timestamp) AS day,
    block_timestamp,
    current_bal
  FROM
    ethereum.core.ez_balance_deltas
  WHERE
    user_address = lower('0xf1dA938Cbf912b9e5444F6532C20A58d09Dd67B8')
),
filled_balances AS (
  SELECT
    ds.day AS day,
    COALESCE(
      bd.current_bal,
      LAST_VALUE(bd.current_bal IGNORE NULLS) OVER (
        ORDER BY
          ds.day ROWS BETWEEN UNBOUNDED PRECEDING
          AND CURRENT ROW
      )
    ) AS filled_balance,
    ROW_NUMBER() OVER (
      PARTITION BY ds.day
      ORDER BY
        bd.block_timestamp DESC
    ) AS rn
  FROM
    date_series ds
    LEFT JOIN balance_data bd ON ds.day = bd.day
),
assets AS (
  SELECT
    fb.day,
    fb.filled_balance * ep.avg_price AS balance_in_usd
  FROM
    filled_balances fb
    LEFT JOIN eth_price_daily ep ON fb.day = ep.day
  WHERE
    fb.rn = 1
),
equity_sales AS (
  SELECT
    date_trunc('day', block_timestamp) AS day,
    ethereum.public.udf_hex_to_int(SUBSTR(DATA, 1, 66)) AS ID,
    ethereum.public.udf_hex_to_int(substr(data, 131, 64)) / power(10, 18) AS amount_eth
  FROM
    ethereum.core.fact_event_logs
  WHERE
    contract_address = lower('0x93519f3558775BBd5c0d501A2fD7a58bb034B379')
    AND topics [0] = lower(
      '0xc9f72b276a388619c6d185d146697036241880c36654b1a3ffdad07c24038d99'
    )
),
daily_data AS (
  SELECT
    ds.day,
    COALESCE(es.transaction_value_eth, 0) AS transaction_value_eth,
    COALESCE(es.unique_units, 0) AS unique_units
  FROM
    date_series ds
    LEFT JOIN (
      SELECT
        day,
        SUM(amount_eth) AS transaction_value_eth,
        COUNT(DISTINCT ID) AS unique_units
      FROM
        equity_sales
      GROUP BY
        day
    ) es ON ds.day = es.day
),
rolling_avg_price AS (
  SELECT
    day,
    AVG(transaction_value_eth / NULLIF(unique_units, 0)) OVER (
      ORDER BY
        day ROWS BETWEEN 29 PRECEDING
        AND CURRENT ROW
    ) AS avg_price_last_30_days
  FROM
    daily_data
),
cumulative_units AS (
  SELECT
    day,
    SUM(unique_units) OVER (
      ORDER BY
        day
    ) AS cumulative_units
  FROM
    daily_data
),
market_cap AS (
  SELECT
    rap.day,
    COALESCE(
      rap.avg_price_last_30_days * ep.avg_price,
      LAST_VALUE(
        rap.avg_price_last_30_days * ep.avg_price IGNORE NULLS
      ) OVER (
        ORDER BY
          rap.day ROWS BETWEEN UNBOUNDED PRECEDING
          AND CURRENT ROW
      )
    ) AS avg_price_last_30_days_usd,
    cu.cumulative_units,
    -1 * COALESCE(
      rap.avg_price_last_30_days * cu.cumulative_units * ep.avg_price,
      LAST_VALUE(
        rap.avg_price_last_30_days * cu.cumulative_units * ep.avg_price IGNORE NULLS
      ) OVER (
        ORDER BY
          rap.day ROWS BETWEEN UNBOUNDED PRECEDING
          AND CURRENT ROW
      )
    ) AS market_capitalization_usd
  FROM
    rolling_avg_price rap
    LEFT JOIN eth_price_daily ep ON rap.day = ep.day
    JOIN cumulative_units cu ON rap.day = cu.day
)
SELECT
  a.day,
  a.balance_in_usd as assets,
  mc.avg_price_last_30_days_usd,
  mc.cumulative_units,
  mc.market_capitalization_usd as equity,
  - mc.market_capitalization_usd as market_cap
FROM
  assets a
  JOIN market_cap mc ON a.day = mc.day
ORDER BY
  a.day desc;


"""

income_statement_sql = """

WITH auctions AS (
  SELECT
    date_trunc('Month', BLOCK_TIMESTAMP) AS month,
    SUM(
      ethereum.public.udf_hex_to_int(substr(data, 131, 64)) / power(10, 18)
    ) AS auction_revenue_eth
  FROM
    ethereum.core.fact_event_logs
  WHERE
    contract_address = lower('0x93519f3558775BBd5c0d501A2fD7a58bb034B379')
    AND topics [0] = lower(
      '0xc9f72b276a388619c6d185d146697036241880c36654b1a3ffdad07c24038d99'
    )
  GROUP BY
    date_trunc('Month', BLOCK_TIMESTAMP)
),
seed_funding AS (
  SELECT
    date_trunc('Month', BLOCK_TIMESTAMP) AS month,
    SUM(VALUE) AS eth_value
  FROM
    ethereum.core.fact_traces
  WHERE
    TO_ADDRESS = lower('0xf1dA938Cbf912b9e5444F6532C20A58d09Dd67B8')
  GROUP BY
    date_trunc('Month', BLOCK_TIMESTAMP)
),
eth_price AS (
  SELECT
    date_trunc('Month', HOUR) AS month,
    avg(price) AS avg_price
  FROM
    ethereum.price.ez_prices_hourly
  WHERE
    symbol = 'WETH'
  GROUP BY
    date_trunc('Month', HOUR)
),
combined_revenue AS (
  SELECT
    sf.month,
    sf.eth_value AS total_revenue_eth,
    a.auction_revenue_eth,
    (
      sf.eth_value - COALESCE(a.auction_revenue_eth, 0)
    ) AS non_auction_revenue_eth,
    sf.eth_value * ep.avg_price AS total_revenue_usd,
    COALESCE(a.auction_revenue_eth, 0) * ep.avg_price AS auction_revenue_usd,
    (
      sf.eth_value - COALESCE(a.auction_revenue_eth, 0)
    ) * ep.avg_price AS non_auction_revenue_usd
  FROM
    seed_funding sf
    LEFT JOIN auctions a ON sf.month = a.month
    LEFT JOIN eth_price ep ON sf.month = ep.month
),
expenses AS (
  SELECT
    date_trunc('Month', BLOCK_TIMESTAMP) AS month,
    - SUM(
      ethereum.public.udf_hex_to_int(substr(data, 451, 64)) / power(10, 18)
    ) AS expenses_eth
  FROM
    ethereum.core.fact_event_logs
  WHERE
    contract_address = lower('0xf1dA938Cbf912b9e5444F6532C20A58d09Dd67B8')
    AND topics [0] = lower(
      '0x7e74d8579043af873f575ed17043a48d6beba2668c6b53325bcd8c9a550e5e9c'
    )
  GROUP BY
    date_trunc('Month', BLOCK_TIMESTAMP)
),
monthly_expenses AS (
  SELECT
    e.month,
    e.expenses_eth,
    e.expenses_eth * ep.avg_price AS expenses_usd
  FROM
    expenses e
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
FROM
  combined_revenue cr FULL
  OUTER JOIN monthly_expenses me ON cr.month = me.month
ORDER BY
  cr.month DESC;


"""