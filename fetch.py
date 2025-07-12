import os
import requests
from datetime import datetime
from zipfile import ZipFile
import io
import pandas as pd

def generate_monthly_urls(symbol, start_date, end_date):
    current = start_date
    urls = []
    while current <= end_date:
        url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1m/{symbol}-1m-{current.strftime('%Y-%m')}.zip"
        urls.append((url, current.strftime('%Y-%m')))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)
    return urls

def download_and_extract_zip(url):
    print(f"下載 {url} ...")
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        print(f"下載失敗，狀態碼 {r.status_code}")
        return None
    with ZipFile(io.BytesIO(r.content)) as z:
        name_list = z.namelist()
        if len(name_list) == 0:
            print("壓縮包內沒有檔案")
            return None
        with z.open(name_list[0]) as f:
            df = pd.read_csv(f)  # 自動判斷header
    return df

def main():
    symbol = "ETHUSDT"
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 6, 1)

    urls = generate_monthly_urls(symbol, start_date, end_date)
    all_dfs = []

    for url, ym in urls:
        df = download_and_extract_zip(url)
        if df is not None and not df.empty:
            print(f"{ym} 資料樣本:")
            print(df.head())
            all_dfs.append(df)
        else:
            print(f"{ym} 資料下載失敗或為空，跳過。")

    if not all_dfs:
        print("沒有任何資料被下載，程式結束。")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 檢查欄位名稱是否存在
    if 'open_time' in combined_df.columns:
        try:
            combined_df['open_time'] = pd.to_datetime(combined_df['open_time'], unit='ms')
        except Exception as e:
            print(f"轉換時間欄位錯誤: {e}")
    else:
        print("找不到 'open_time' 欄位，請確認資料格式。")

    combined_df = combined_df.sort_values('open_time').reset_index(drop=True)

    print(f"總共合併 {len(combined_df)} 筆資料。")

    output_file = f'{symbol}_1m_klines_merged.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"合併後CSV檔案已儲存：{output_file}")

if __name__ == "__main__":
    main()
