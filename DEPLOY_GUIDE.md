# 🚀 HƯỚNG DẪN DEPLOY NHANH

## Bước 1: Chuẩn bị files

✅ Đã có tất cả files cần thiết:
- app.py (Flask API)
- requirements.txt (Dependencies)
- Procfile (Deploy config)
- .gitignore
- README.md (Tài liệu đầy đủ)
- demo.html (Test UI)
- test_api.py (Test script)
- test_request.json (Sample data)

⚠️ BẠN CẦN THÊM:
- football_model_v2.joblib (file model đã train)

## Bước 2: Test Local

```bash
# Cài dependencies
pip install -r requirements.txt

# Copy file model vào thư mục
# cp /path/to/football_model_v2.joblib ./

# Chạy API
python app.py

# Mở demo.html trong browser
# Hoặc chạy test script:
python test_api.py
```

## Bước 3: Push lên GitHub

```bash
git init
git add .
git commit -m "Initial commit - Football Prediction API"

# Tạo repo trên GitHub
# Sau đó:
git remote add origin https://github.com/YOUR_USERNAME/football-predictor.git
git branch -M main
git push -u origin main
```

## Bước 4: Deploy lên Render

1. Vào https://render.com
2. Sign up (miễn phí)
3. New → Web Service
4. Connect GitHub repository vừa tạo
5. Settings:
   - Name: football-predictor
   - Environment: Python 3
   - Build Command: pip install -r requirements.txt
   - Start Command: gunicorn app:app
6. Click "Create Web Service"
7. Đợi 3-5 phút

✅ Done! API sẽ online tại: https://football-predictor.onrender.com

## Bước 5: Test API đã deploy

Mở demo.html, thay API URL thành:
```
https://your-app-name.onrender.com
```

Hoặc test bằng curl:
```bash
curl https://your-app-name.onrender.com/health
```

## ⚠️ Nếu model file quá lớn (>100MB)

### Option A: Git LFS
```bash
git lfs install
git lfs track "*.joblib"
git add .gitattributes
git commit -m "Track model with LFS"
```

### Option B: Google Drive
1. Upload football_model_v2.joblib lên Google Drive
2. Share → Anyone with link can view
3. Copy File ID từ link
4. Sửa app.py, thêm đầu file:

```python
import requests
import os

MODEL_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"

if not os.path.exists('football_model_v2.joblib'):
    print("Downloading model...")
    r = requests.get(MODEL_URL, allow_redirects=True)
    open('football_model_v2.joblib', 'wb').write(r.content)
```

5. Thêm vào requirements.txt:
```
requests==2.31.0
```

## 📞 Support

Nếu gặp vấn đề:
1. Check Render Logs
2. Test local trước
3. Đảm bảo requirements.txt đúng version
4. Model file phải có trong repo hoặc download được

## 🎯 Next Steps

Sau khi deploy thành công:
- Integrate vào web frontend
- Thêm authentication (API key)
- Setup monitoring
- Thêm caching cho performance
- Collect feedback và retrain model
