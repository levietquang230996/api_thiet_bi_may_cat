# PMIS Device Suggestion System

Hệ thống gợi ý thiết bị điện cho PMIS (Power Management Information System).

## Cấu trúc dự án

```
PMIS v 13/
├── api/                    # API FastAPI
│   ├── __init__.py
│   └── app.py              # Main API application
├── config/                 # Cấu hình
│   ├── __init__.py
│   └── settings.py         # Settings và constants
├── data/                   # Dữ liệu
│   ├── devicesPMISMayCat.csv           # Dữ liệu gốc
│   └── devicesPMISMayCat_cleaned.csv   # Dữ liệu đã làm sạch
├── logs/                   # Log files
├── models/                 # Model artifacts
├── notebooks/              # Jupyter notebooks
│   ├── 01_preprocess_devices.ipynb     # Tiền xử lý dữ liệu
│   └── 02_train_model.ipynb            # Huấn luyện model
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── data_utils.py       # Data processing utilities
│   └── model_utils.py      # Model utilities
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## Cài đặt

### 1. Tạo môi trường ảo

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Tiền xử lý dữ liệu

Chạy notebook `notebooks/01_preprocess_devices.ipynb` để:
- Đọc và phân tích dữ liệu
- Làm sạch dữ liệu
- Áp dụng quy tắc chuẩn hóa
- Xuất file `devicesPMISMayCat_cleaned.csv`

### 2. Huấn luyện model

Chạy notebook `notebooks/02_train_model.ipynb` để:
- Chuẩn bị dữ liệu
- Xây dựng và huấn luyện model
- Đánh giá hiệu năng
- Xuất model artifacts

### 3. Chạy API

```bash
cd api
python app.py
```

Hoặc với uvicorn:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

API documentation có tại:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health Check
```
GET /api/v1/health
```

### Gợi ý thiết bị (không score)
```
POST /api/v1/suggest
```

Input: Các trường ID (không có `_DESC`)
```json
{
  "P_MANUFACTURERID": "HSX.00311",
  "NATIONALFACT": "TB040.00006",
  "top_n": 5
}
```

### Gợi ý thiết bị (có score)
```
POST /api/v1/suggest-with-score
```

Input: Các trường ID (không có `_DESC`)
Output: Danh sách thiết bị + confidence score

### Gợi ý từ OCR
```
POST /api/v1/suggest-from-ocr
```

Input: Các trường `_DESC` từ OCR
```json
{
  "ASSETDESC": "Máy cắt 171",
  "P_MANUFACTURERID_DESC": "Siemens",
  "top_n": 5
}
```

## Quy tắc chuẩn hóa (bắt buộc)

| Cột | Giá trị chuẩn |
|-----|---------------|
| PHA | EVN.PHA_3P |
| KIEU_MC | TBI_CT_MC_KIEU_MC_01 |
| KIEU_DAPHQ | TBI_TT_MC_KIEU_DAPHQ.00001 |
| KIEU_CD | TBI_CT_MC_CC_CD.00001 |
| U_TT | TBI_CT_MC_U_TT_02 |
| NATIONALFACT | Không được là TB040.00023 |

## Model Metrics

- Accuracy: ~0.85+
- Precision: ~0.80+
- Recall: ~0.80+
- F1-Score: ~0.80+
- Latency: < 10ms/request

## Phát triển

### Chạy tests

```bash
pytest tests/
```

### Format code

```bash
# Cài đặt dev tools
pip install black isort flake8

# Format
black .
isort .
flake8 .
```

## Giấy phép

Copyright (c) 2025 PMIS Team. All rights reserved.
