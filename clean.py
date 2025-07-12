import pandas as pd
import numpy as np

def clean_and_label_optimized(df,
                               price_col='close',
                               high_col='high',
                               low_col='low',
                               future_window=120,
                               up_threshold=0.05,
                               max_drawdown=-0.01):
    df = df.copy()

    # 時間處理
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.sort_values('open_time').reset_index(drop=True)

    # 去除重複與空值
    df = df.drop_duplicates(subset=['open_time'])
    df = df.dropna(subset=[price_col, high_col, low_col])

    # numpy 陣列加速
    close_prices = df[price_col].values
    high_prices = df[high_col].values
    low_prices = df[low_col].values

    n = len(df)
    labels = np.zeros(n)
    future_returns = np.full(n, np.nan)
    future_drawdowns = np.full(n, np.nan)

    for i in range(n - future_window):
        current_close = close_prices[i]
        future_highs = high_prices[i+1:i+1+future_window]
        future_lows = low_prices[i+1:i+1+future_window]

        max_high = np.max(future_highs)
        min_low = np.min(future_lows)

        future_return = (max_high - current_close) / current_close
        future_drawdown = (min_low - current_close) / current_close

        future_returns[i] = future_return
        future_drawdowns[i] = future_drawdown

        if (future_return >= up_threshold) and (future_drawdown >= max_drawdown):
            labels[i] = 1

    # 寫入 DataFrame
    df['future_return'] = future_returns
    df['future_drawdown'] = future_drawdowns
    df['label'] = labels.astype(int)

    return df

# === 使用方式 ===
file_path = 'ETHUSDT_1m_klines_merged.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.lower().str.strip()

labeled_df = clean_and_label_optimized(df)

# 顯示尾部
print(labeled_df[['open_time', 'close', 'future_return', 'future_drawdown', 'label']].tail(10))
