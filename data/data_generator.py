"""
Data Generator & Loader cho dự án Inventory Management RL
==========================================================
- Generate synthetic demand data (Poisson, Seasonal, Trending)
- Tải và xử lý real-world dataset
- Xuất CSV để sử dụng trong training
"""

import numpy as np
import pandas as pd
import os
from typing import Optional


def generate_seasonal_demand(
    n_days: int = 730,
    base_demand: float = 30.0,
    seasonal_amplitude: float = 15.0,
    weekly_amplitude: float = 8.0,
    noise_std: float = 5.0,
    trend: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tạo dữ liệu nhu cầu hàng hóa theo mùa (realistic).
    
    Mô phỏng nhu cầu hàng hóa thực tế với:
    - Xu hướng tăng/giảm dần (trend)
    - Biến động theo mùa (seasonal - chu kỳ năm)
    - Biến động theo tuần (weekly - cuối tuần cao hơn)
    - Nhiễu ngẫu nhiên (noise)
    - Sự kiện đặc biệt (promotions, holidays)
    
    Args:
        n_days: Số ngày dữ liệu
        base_demand: Nhu cầu cơ bản trung bình / ngày
        seasonal_amplitude: Biên độ biến động mùa
        weekly_amplitude: Biên độ biến động tuần
        noise_std: Độ lệch chuẩn nhiễu
        trend: Hệ số xu hướng tăng
        seed: Random seed
        
    Returns:
        DataFrame với columns: date, demand, day_of_week, month, is_weekend, is_holiday
    """
    np.random.seed(seed)
    
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")
    days = np.arange(n_days)
    
    # 1. Base demand + trend
    base = base_demand + trend * days
    
    # 2. Seasonal component (chu kỳ năm - mùa hè demand cao hơn)
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * days / 365 + np.pi / 2)
    
    # 3. Weekly component (cuối tuần demand cao hơn)
    weekly = weekly_amplitude * np.sin(2 * np.pi * days / 7)
    
    # 4. Random noise
    noise = np.random.normal(0, noise_std, n_days)
    
    # 5. Special events (Black Friday, Tết, promotions)
    events = np.zeros(n_days)
    for i in range(n_days):
        date = dates[i]
        # Tết Nguyên Đán (khoảng cuối tháng 1 - đầu tháng 2)
        if date.month == 1 and date.day >= 20:
            events[i] = 25
        if date.month == 2 and date.day <= 10:
            events[i] = 20
        # Black Friday (cuối tháng 11)
        if date.month == 11 and date.day >= 25:
            events[i] = 30
        # Giáng sinh
        if date.month == 12 and date.day >= 20:
            events[i] = 20
        # Sale tháng 6 (mid-year sale)
        if date.month == 6 and 15 <= date.day <= 20:
            events[i] = 15
    
    # Tổng hợp demand
    demand = base + seasonal + weekly + noise + events
    demand = np.maximum(0, np.round(demand)).astype(int)
    
    # Tạo DataFrame
    df = pd.DataFrame({
        "date": dates,
        "demand": demand,
        "day_of_week": [d.dayofweek for d in dates],
        "day_name": [d.strftime("%A") for d in dates],
        "month": [d.month for d in dates],
        "is_weekend": [1 if d.dayofweek >= 5 else 0 for d in dates],
        "is_holiday": [1 if events[i] > 0 else 0 for i, d in enumerate(dates)],
        "base_demand": np.round(base, 1),
        "seasonal_component": np.round(seasonal, 1),
        "weekly_component": np.round(weekly, 1),
    })
    
    return df


def generate_multi_product_demand(
    n_days: int = 365,
    n_products: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tạo dữ liệu nhu cầu cho nhiều sản phẩm.
    
    Args:
        n_days: Số ngày
        n_products: Số sản phẩm
        seed: Random seed
        
    Returns:
        DataFrame với demand cho từng sản phẩm
    """
    np.random.seed(seed)
    
    product_configs = [
        {"name": "Gạo (kg)", "base": 50, "seasonal_amp": 10, "weekly_amp": 5},
        {"name": "Nước ngọt (thùng)", "base": 30, "seasonal_amp": 20, "weekly_amp": 10},
        {"name": "Mì gói (thùng)", "base": 40, "seasonal_amp": 8, "weekly_amp": 3},
        {"name": "Dầu ăn (chai)", "base": 20, "seasonal_amp": 5, "weekly_amp": 2},
        {"name": "Sữa (thùng)", "base": 35, "seasonal_amp": 12, "weekly_amp": 8},
    ]
    
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")
    data = {"date": dates}
    
    for i in range(min(n_products, len(product_configs))):
        cfg = product_configs[i]
        days = np.arange(n_days)
        demand = (
            cfg["base"]
            + cfg["seasonal_amp"] * np.sin(2 * np.pi * days / 365)
            + cfg["weekly_amp"] * np.sin(2 * np.pi * days / 7)
            + np.random.normal(0, 3, n_days)
        )
        data[cfg["name"]] = np.maximum(0, np.round(demand)).astype(int)
    
    return pd.DataFrame(data)


def generate_realistic_warehouse_data(
    n_days: int = 730,
    seed: int = 42,
) -> dict:
    """
    Tạo bộ dữ liệu warehouse hoàn chỉnh bao gồm:
    - Demand data
    - Supplier info
    - Product catalog
    - Cost parameters
    
    Returns:
        Dictionary chứa nhiều DataFrames
    """
    np.random.seed(seed)
    
    # 1. Product Catalog
    products = pd.DataFrame({
        "product_id": [f"P{i:03d}" for i in range(1, 11)],
        "product_name": [
            "Gạo ST25 (kg)", "Nước suối Aqua (thùng)", "Mì Hảo Hảo (thùng)",
            "Dầu Neptune (chai)", "Sữa Vinamilk (thùng)", "Bia Saigon (thùng)",
            "Bột giặt OMO (bịch)", "Nước mắm Chinsu (chai)", "Cà phê G7 (hộp)",
            "Bánh Orion (thùng)"
        ],
        "unit_cost": [15, 80, 65, 45, 120, 200, 55, 25, 90, 110],
        "holding_cost_per_unit": [0.5, 1.0, 0.8, 0.6, 1.5, 2.0, 0.7, 0.4, 1.0, 1.2],
        "stockout_cost_per_unit": [3.0, 5.0, 4.0, 3.0, 7.0, 10.0, 3.5, 2.0, 5.0, 6.0],
        "max_storage": [500, 200, 300, 200, 150, 100, 250, 300, 200, 150],
        "shelf_life_days": [365, 180, 240, 365, 30, 180, 365, 365, 180, 120],
        "category": [
            "Lương thực", "Nước uống", "Lương thực", "Gia vị", "Sữa",
            "Nước uống", "Tẩy rửa", "Gia vị", "Nước uống", "Bánh kẹo"
        ]
    })
    
    # 2. Supplier Info
    suppliers = pd.DataFrame({
        "supplier_id": [f"S{i:02d}" for i in range(1, 6)],
        "supplier_name": [
            "Công ty Lương thực Miền Nam",
            "Đại lý nước giải khát Sài Gòn",
            "Nhà phân phối Acecook",
            "Công ty Dầu thực vật",
            "Vinamilk Distribution"
        ],
        "lead_time_days": [2, 1, 3, 2, 1],
        "fixed_order_cost": [50, 30, 40, 35, 45],
        "min_order_qty": [50, 20, 30, 20, 10],
        "reliability": [0.95, 0.98, 0.92, 0.96, 0.99],
    })
    
    # 3. Demand Data cho sản phẩm chính (Nước suối)
    demand_df = generate_seasonal_demand(
        n_days=n_days,
        base_demand=30,
        seasonal_amplitude=15,
        weekly_amplitude=8,
        noise_std=5,
        seed=seed,
    )
    
    # 4. Multi-product demand
    multi_demand = generate_multi_product_demand(n_days=min(n_days, 365), seed=seed)
    
    return {
        "products": products,
        "suppliers": suppliers,
        "demand": demand_df,
        "multi_product_demand": multi_demand,
    }


def save_data(data_dir: str = "data", n_days: int = 730, seed: int = 42):
    """Tạo và lưu toàn bộ dữ liệu vào thư mục data/."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate data
    warehouse_data = generate_realistic_warehouse_data(n_days=n_days, seed=seed)
    
    # Save to CSV
    warehouse_data["demand"].to_csv(
        os.path.join(data_dir, "demand_data.csv"), index=False, encoding="utf-8-sig"
    )
    warehouse_data["products"].to_csv(
        os.path.join(data_dir, "product_catalog.csv"), index=False, encoding="utf-8-sig"
    )
    warehouse_data["suppliers"].to_csv(
        os.path.join(data_dir, "supplier_info.csv"), index=False, encoding="utf-8-sig"
    )
    warehouse_data["multi_product_demand"].to_csv(
        os.path.join(data_dir, "multi_product_demand.csv"), index=False, encoding="utf-8-sig"
    )
    
    print(f"✅ Đã lưu dữ liệu vào thư mục '{data_dir}/':")
    print(f"   - demand_data.csv ({len(warehouse_data['demand'])} rows)")
    print(f"   - product_catalog.csv ({len(warehouse_data['products'])} products)")
    print(f"   - supplier_info.csv ({len(warehouse_data['suppliers'])} suppliers)")
    print(f"   - multi_product_demand.csv ({len(warehouse_data['multi_product_demand'])} rows)")
    
    return warehouse_data


if __name__ == "__main__":
    save_data()
