# Hướng dẫn Push Code lên GitHub

## Bước 1: Mở Terminal/PowerShell trong thư mục project

Mở PowerShell hoặc Command Prompt tại thư mục project:
```
D:\Work\Projects\Đề tài\2025\PMIS MayCat
```

## Bước 2: Khởi tạo Git (nếu chưa có)

```bash
git init
```

## Bước 3: Thêm remote repository

```bash
git remote add origin https://github.com/levietquang230996/api_thiet_bi_may_cat.git
```

Nếu remote đã tồn tại, dùng:
```bash
git remote set-url origin https://github.com/levietquang230996/api_thiet_bi_may_cat.git
```

## Bước 4: Kiểm tra các file cần commit

```bash
git status
```

## Bước 5: Thêm tất cả các file (QUAN TRỌNG: Bao gồm thư mục model/)

```bash
# Thêm file .gitignore trước
git add .gitignore

# Thêm tất cả các file khác (bao gồm model/)
git add .

# Kiểm tra lại xem model/ đã được thêm chưa
git status
```

**LƯU Ý QUAN TRỌNG**: Bạn PHẢI thấy các file `.pkl` trong thư mục `model/` được list trong `git status`. Nếu không thấy, cần kiểm tra `.gitignore` và đảm bảo dòng `# model/*.pkl` vẫn đang bị comment.

## Bước 6: Commit code

```bash
git commit -m "Initial commit: Add FastAPI project with models"
```

## Bước 7: Push lên GitHub

```bash
# Lần đầu push (set upstream)
git push -u origin main
```

Hoặc nếu branch của bạn là `master`:
```bash
git push -u origin master
```

## Bước 8: Kiểm tra trên GitHub

1. Truy cập: https://github.com/levietquang230996/api_thiet_bi_may_cat
2. Kiểm tra xem code đã được push chưa
3. Đặc biệt kiểm tra thư mục `model/` có các file `.pkl` chưa

## Troubleshooting

### Nếu gặp lỗi "fatal: not a git repository"
- Đảm bảo bạn đang ở đúng thư mục project
- Chạy lại `git init`

### Nếu model/*.pkl không được add
- Kiểm tra file `.gitignore`
- Đảm bảo dòng `# model/*.pkl` vẫn đang bị comment (có dấu #)
- Nếu bị uncomment (không có #), comment lại bằng cách thêm # ở đầu

### Nếu gặp lỗi authentication
- GitHub hiện tại yêu cầu Personal Access Token (PAT)
- Tạo token tại: https://github.com/settings/tokens
- Khi push, dùng token thay vì password

## Sau khi push thành công

Bạn sẽ có thể:
1. ✅ Thấy code trên GitHub
2. ✅ Thấy thư mục `model/` với các file `.pkl`
3. ✅ Deploy lên Render (xem file `DEPLOY.md`)

## Tiếp theo: Deploy lên Render

Sau khi code đã được push lên GitHub thành công, làm theo hướng dẫn trong file `DEPLOY.md` để deploy lên Render.
