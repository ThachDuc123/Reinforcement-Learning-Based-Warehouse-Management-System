import os
import pandas as pd
import numpy as np
import requests

def download_online_retail_data(data_dir: str = "data") -> str:
    """
    Download public dataset: Online Retail Data Set from UCI Machine Learning Repository
    URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
    """
    os.makedirs(data_dir, exist_ok=True)
    excel_path = os.path.join(data_dir, "online_retail.xlsx")
    
    if not os.path.exists(excel_path):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        print(f"Bắt đầu tải dữ liệu thật (Online Retail) từ {url} ...")
        response = requests.get(url, stream=True)
        with open(excel_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Tải xong file Excel!")
    else:
        print("File Excel đã tồn tại, bỏ qua bước download.")
        
    return excel_path

def process_retail_data_to_demand(excel_path: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Đọc file excel online retail, gom nhóm Quantity theo ngày để tạo series daily demand.
    Chúng ta sẽ lấy lịch sử daily total demand để làm data cho một 'tổng kho' nhé.
    """
    print("Đang đọc file Excel (có thể mất vài phút)...")
    df = pd.read_excel(excel_path)
    
    # Xử lý data cơ bản
    # Chỉ lấy các dòng có số lượng dương (mua hàng, không tính trả lại)
    df = df[df['Quantity'] > 0]
    
    # Rút trích thông tin ngày từ dấu thời gian
    df['date'] = df['InvoiceDate'].dt.date
    
    # Gom nhóm theo ngày để tính tổng demand
    daily_demand = df.groupby('date')['Quantity'].sum().reset_index()
    daily_demand.rename(columns={'Quantity': 'demand'}, inplace=True)
    daily_demand['date'] = pd.to_datetime(daily_demand['date'])
    
    # Điền giá trị 0 cho những ngày không có đơn nào
    min_date = daily_demand['date'].min()
    max_date = daily_demand['date'].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    
    daily_demand = daily_demand.set_index('date').reindex(all_dates, fill_value=0).reset_index()
    daily_demand.rename(columns={'date': 'date_old'}, inplace=True) # Workaround to rename index properly
    daily_demand.rename(columns={'index': 'date'}, inplace=True)
    
    # Bổ sung các thông tin khác (thứ, tháng...)
    daily_demand['day_of_week'] = daily_demand['date'].dt.dayofweek
    daily_demand['month'] = daily_demand['date'].dt.month
    
    # Vì daily total volume (tổng ngày) có thể là con số quá lớn cho cái kho trong environment của ta (max inventory 200). 
    # Ta sẽ scale down con số này, gán như là một kho hàng nhỏ hoặc một sản phẩm tiêu biểu (ví dụ: demand = demand / 500)
    # Ta chuẩn hóa sao cho demand trung bình rơi vào khoảng 30 (bằng với mean_demand của environment).
    
    mean_real_demand = daily_demand['demand'].mean()
    scale_factor = mean_real_demand / 30.0
    
    daily_demand['demand'] = np.round(daily_demand['demand'] / scale_factor).astype(int)
    
    # Lưu xuống CSV
    csv_path = os.path.join(data_dir, "real_daily_demand.csv")
    daily_demand.to_csv(csv_path, index=False)
    
    print(f"Đã xử lý và lưu demand data vào '{csv_path}'.")
    print(f"Tổng số ngày: {len(daily_demand)}")
    print(f"Demand trung bình mới: {daily_demand['demand'].mean():.1f}")
    
    return daily_demand

if __name__ == "__main__":
    filepath = download_online_retail_data()
    process_retail_data_to_demand(filepath)
