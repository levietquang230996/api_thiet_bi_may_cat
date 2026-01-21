# PMIS AutoFill API

API service để predict các trường tự động điền dựa trên model đã train.

## Cài đặt

1. Cài đặt dependencies:
```bash
pip install -r ../requirements.txt
```

Hoặc nếu đang ở thư mục gốc:
```bash
pip install -r requirements.txt
```

## Chạy API

Từ thư mục gốc của project:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Hoặc từ thư mục `api/`:
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Truy cập API

- API Documentation (Swagger UI): http://localhost:8000/docs
- Alternative docs (ReDoc): http://localhost:8000/redoc
- Health check: http://localhost:8000/health

## Endpoints

- `GET /health` - Kiểm tra trạng thái API
- `GET /targets` - Liệt kê các target columns có sẵn
- `POST /predict/{target_col}` - Predict cho một target cụ thể
- `POST /predict/all` - Predict cho tất cả targets

## Ví dụ sử dụng

### Health check
```bash
curl http://localhost:8000/health
```

### List targets
```bash
curl http://localhost:8000/targets
```

### Predict một target
```bash
curl -X POST "http://localhost:8000/predict/CATEGORYID" \
  -H "Content-Type: application/json" \
  -d '{
    "loai": "value1",
    "p_manufacturerid": "value2",
    "nationalfact": "value3"
  }'
```

### Predict tất cả targets
```bash
curl -X POST "http://localhost:8000/predict/all" \
  -H "Content-Type: application/json" \
  -d '{
    "loai": "value1",
    "p_manufacturerid": "value2",
    "nationalfact": "value3"
  }'
```

