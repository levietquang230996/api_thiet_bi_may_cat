# Hướng dẫn Deploy lên Render (Miễn phí)

## Bước 1: Chuẩn bị

1. **Đảm bảo code đã được push lên GitHub**
   - Nếu chưa có repo, tạo repo mới trên GitHub
   - Push toàn bộ code lên GitHub (bao gồm thư mục `model/` với các file `.pkl`)

## Bước 2: Đăng ký Render

1. Truy cập: https://render.com
2. Đăng ký/Đăng nhập bằng tài khoản GitHub (dễ nhất)
3. **KHÔNG CẦN** thẻ tín dụng cho free tier

## Bước 3: Tạo Web Service

1. Vào Dashboard → Click **"New +"** → Chọn **"Web Service"**
2. Kết nối GitHub repository của bạn
3. Chọn repository chứa code API
4. Render sẽ tự động detect các file config:
   - `render.yaml` (nếu có)
   - `Procfile`
   - `requirements.txt`

## Bước 4: Cấu hình (nếu cần)

Render sẽ tự động detect:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: Từ `Procfile` hoặc `render.yaml`

Có thể để mặc định hoặc chỉnh nếu cần.

## Bước 5: Deploy

1. Click **"Create Web Service"**
2. Render sẽ tự động:
   - Install dependencies từ `requirements.txt`
   - Build application
   - Deploy và start service
3. Chờ deploy hoàn tất (có thể mất 5-10 phút lần đầu)

## Bước 6: Kiểm tra

1. Sau khi deploy xong, bạn sẽ nhận được URL: `https://pmis-autofill-api.onrender.com`
2. Test API:
   ```bash
   # Health check
   curl https://your-app-name.onrender.com/health
   
   # API docs
   https://your-app-name.onrender.com/docs
   ```

## Lưu ý quan trọng

### ⚠️ Free Tier Limitations:
- Service sẽ **sleep sau 15 phút** không có request
- Request đầu tiên sau khi sleep sẽ mất **~30 giây** để wake up
- Vẫn hoàn toàn miễn phí và không cần thẻ tín dụng

### 🔧 Giữ service không sleep (tùy chọn):
1. **UptimeRobot** (miễn phí):
   - Đăng ký: https://uptimerobot.com
   - Tạo monitor ping URL của bạn mỗi 5 phút
   - Service sẽ không bao giờ sleep

2. **Cron-job.org** (miễn phí):
   - Tương tự, setup cron job ping API mỗi 5 phút

### 📦 Đảm bảo thư mục model có trong repo:
Render cần file models để chạy. Đảm bảo thư mục `model/` và các file `.pkl` đã được commit vào Git:
```bash
git add model/
git commit -m "Add model files"
git push
```

### 🔍 Troubleshooting:

1. **Lỗi "Module not found"**:
   - Kiểm tra `requirements.txt` có đầy đủ dependencies
   - Xem build logs trong Render dashboard

2. **Lỗi "Bundle not loaded"**:
   - Kiểm tra thư mục `model/` có trong repo
   - Kiểm tra đường dẫn MODEL_DIR trong code

3. **Build timeout**:
   - PyTorch và CatBoost khá nặng, có thể mất thời gian install
   - Kiên nhẫn đợi (5-10 phút là bình thường)

## Kết quả

Sau khi deploy thành công, bạn sẽ có:
- ✅ API chạy 24/7 (miễn phí)
- ✅ URL công khai để gọi API
- ✅ Swagger UI documentation tại `/docs`
- ✅ Không cần thẻ tín dụng

Chúc bạn deploy thành công! 🚀
