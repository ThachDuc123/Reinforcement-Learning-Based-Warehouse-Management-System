# 🏭 Inventory Management with Reinforcement Learning
## Quản Lý Kho Hàng bằng Reinforcement Learning

> **Môn học:** REL301m - Reinforcement Learning

---

## 📝 Mô tả dự án

Dự án xây dựng hệ thống AI sử dụng **Reinforcement Learning** để tự động quyết định:
- **Khi nào** cần nhập hàng vào kho?
- **Bao nhiêu** hàng cần đặt mỗi lần?

Mục tiêu: **Tối thiểu hóa tổng chi phí** bao gồm:
- 💰 Chi phí tồn kho (Holding Cost)
- ❌ Chi phí thiếu hàng (Stockout Cost) 
- 🚚 Chi phí đặt hàng (Ordering Cost)

## 🧠 Thuật toán

| Thuật toán | Loại | Mô tả |
|---|---|---|
| **Q-Learning** | Tabular RL | Q-table với state discretization |
| **DQN** | Deep RL | Neural Network + Experience Replay + Target Network |

### So sánh với Baseline:
- **Random Policy**: Đặt hàng ngẫu nhiên
- **Fixed Order**: Đặt lượng cố định mỗi ngày
- **(s, S) Policy**: Chính sách tồn kho cổ điển
- **EOQ Policy**: Economic Order Quantity

## 📁 Cấu trúc dự án

```
project/
├── main_notebook.ipynb      # 📓 Notebook chính (chạy toàn bộ pipeline)
├── train.py                 # 🏋️ Script huấn luyện
├── requirements.txt         # 📦 Dependencies
├── README.md               
│
├── env/                     # 🎮 Gymnasium Environment
│   ├── __init__.py
│   └── inventory_env.py     # Custom InventoryEnv
│
├── agents/                  # 🤖 RL Agents
│   ├── __init__.py
│   ├── q_learning.py        # Q-Learning Agent
│   ├── dqn_agent.py         # DQN Agent
│   └── baselines.py         # Baseline Policies
│
├── data/                    # 📊 Data
│   ├── __init__.py
│   ├── data_generator.py    # Tạo demand data
│   ├── demand_data.csv      # Demand data (730 ngày)
│   ├── product_catalog.csv  # Danh mục sản phẩm
│   └── supplier_info.csv    # Thông tin nhà cung cấp
│
├── utils/                   # 🛠️ Utilities
│   ├── __init__.py
│   └── visualization.py     # Biểu đồ & charts
│
├── models/                  # 💾 Saved Models
│   ├── q_learning_model.pkl
│   └── dqn_model.pth
│
└── figures/                 # 📈 Saved Figures
    ├── demand_analysis.png
    ├── training_curves.png
    ├── evaluation_comparison.png
    └── ...
```

## 🚀 Cách chạy

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Chạy Notebook (Khuyến nghị)
Mở file `main_notebook.ipynb` trong Jupyter/VS Code và chạy từng cell.

### 3. Hoặc chạy script
```bash
python train.py
```

## 📐 Bài toán RL cho Quản lý kho hàng (MDP Formulation)

Đây là khung Formulation (hợp đồng Agent - Môi trường) dùng cho báo cáo:

### 1) State ($s_t$) — Trạng thái tại ngày $t$
Vector 4 chiều bao gồm:
1. `inventory_level_norm`: Tồn kho hiện tại đã chuẩn hóa về `[-1, 1]` hoặc `[0, 1]`.
2. `day_of_week_norm`: Ngày trong tuần `(0-6) / 6` (để học chu kỳ tuần).
3. `pending_orders_norm`: Tổng hàng đang trên đường về kho, chuẩn hóa theo Max Order.
4. `last_demand_norm`: Nhu cầu thực tế của ngày hôm trước để đoán xu hướng.

👉 **Ý nghĩa:** Agent nhìn được "Trong kho còn bao nhiêu, hôm nay thứ mấy, hàng nào chưa về và hôm qua khách mua bao nhiêu".

### 2) Action ($a_t$) — Quyết định nhập hàng
Action rời rạc (Discrete): $a_t \in \{0, 1, 2, ..., N-1\}$
- Số lượng đặt thực tế (Order Quantity) $= a_t \times \text{order\_step}$
- *Ví dụ:* Nếu $max\_order = 100$, $order\_step = 10$ thì Action tưng ứng mức đặt: $\{0, 10, 20, 30, \dots, 100\}$ đơn vị.

### 3) Transition — Cập nhật môi trường hàng ngày
Thứ tự cập nhật trong 1 Step:
1. Agent chọn mức lượng đặt hàng (Order Qty).
2. Nếu có đặt, thêm vào hàng chờ giao (Pending Order với khoảng trễ `lead_time`).
3. Nhận hàng đã đến kỳ hạn giao cộng dồn vào `inventory`.
4. Phát sinh nhu cầu khách hàng ($d_t$).
5. Tính `stockout = max(0, d_t - inventory)`.
6. Tính `inventory` thực còn lại sau bán: `max(0, inventory - d_t)`.

### 4) Reward ($r_t$) — Hàm Thưởng/Phạt
Reward tỉ lệ nghịch (âm) với tổng chi phí ngày đó:
$$r_t = -[c_h \cdot I_t + c_s \cdot SO_t + \mathbb{1}_{(q_t > 0)} \cdot c_f + c_u \cdot q_t]$$
**Trong đó:**
- $I_t$: Tồn kho cuối ngày (Inventory)
- $SO_t$: Số lượng thiếu hàng (Stockout)
- $q_t$: Lượng đặt hàng ngày t (Order Quantity)
- $c_h$: Chi phí lưu kho (Holding cost)
- $c_s$: Chi phí thiếu/hết hàng (Stockout penalty cost)
- $c_f$: Tiền cố định mỗi đơn đặt hàng (Fixed order cost)
- $c_u$: Chi phí đơn vị đặt hàng (Unit cost)

👉 **Mục tiêu:** $\max \sum r_t \Leftrightarrow \min \sum (\text{Tổng Chi phí})$

### 5) Done (Episode Termination)
Kết thúc 1 Episode khi bước thời gian (day) đạt ngưỡng `episode_length` (ví dụ: 365 ngày).

## 🚀 Định hướng Nâng Cấp & Ứng dụng Thực tế

Nếu phát triển tiếp, dự án có thể mở rộng theo các hướng sau (rất phù hợp để đưa vào kết luận báo cáo):

### 1. Nâng cấp mô hình RL (Model Upgrades)
- **Sử dụng PPO (Proximal Policy Optimization):** Thay vì Action rời rạc như DQN, PPO giúp agent đưa ra con số đặt hàng chính xác dưới dạng số thực (Continuous Action Space).
- **Double DQN / Dueling DQN:** Khắc phục nhược điểm "Overestimation" (Đánh giá quá cao Reward) của mạng DQN truyền thống, giúp Agent học ra chính sách ổn định hơn.
- **Tích hợp LSTM:** Xâu chuỗi lượng Demand thực tế của 7-14 ngày trước đó qua mạng LSTM trước khi đưa vào Agent để Model tự động "Forecast" (dự báo) được chu kì mùa vụ.

### 2. Ý nghĩa kinh doanh & Hiệu quả Tối ưu (Business Value)
- **Tự động hóa hoàn toàn:** Loại bỏ công việc kiểm đếm và tính toán công thức đặt hàng thủ công (EOQ). Mô hình tự động ra quyết định hàng ngày dựa trên Data thời gian thực.
- **Tối ưu Vốn lưu động (Holding Cost):** Mô hình tự nhận biết thời gian chờ (Lead Time) để gọi hàng "Vừa kịp lúc" (Just-in-time), tránh găm hàng thừa gây đọng vốn.
- **Chống Đứt gãy Chuỗi cung ứng (Stockout Cost):** Agent nhận diện được yếu tố "Ngày trong tuần" để tự động nhập lượng hàng lớn trước cuối tuần hoặc dịp lễ, đảm bảo doanh thu bán hàng.
- **Gộp đơn thông minh (Order Cost):** Tự động học thói quen dồn đơn (batching) để giảm thiểu chi phí cố định (tiền xe vận tải) mỗi lần gọi hàng từ nhà cung cấp.

## 📊 Dataset

Dữ liệu hỗ trợ 2 nguồn để phục vụ việc train mô hình:
1. **Synthetic data:** 730 ngày demand với biến động Mùa (Season), Tuần, Sự kiện (Sales)
