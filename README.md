# Football Prediction API

API dự đoán kết quả bóng đá sử dụng Machine Learning (Random Forest + Gradient Boosting Ensemble).

## 🎯 Features

- **1x2 Prediction**: Dự đoán Win/Draw/Loss với xác suất
- **Over/Under 2.5**: Dự đoán tổng số bàn thắng
- **Score Prediction**: Dự đoán tỉ số cụ thể
- **2 Endpoints**: Full features hoặc simplified input

## 📁 Project Structure

```
football-predictor/
├── app.py                      # Flask API
├── football_model_v2.joblib    # Model đã train (cần copy vào)
├── requirements.txt            # Dependencies
├── Procfile                    # Deploy config
├── .gitignore
└── README.md
```

## 🚀 Quick Start - Local

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Copy model file

```bash
# Copy file football_model_v2.joblib vào thư mục này
```

### 3. Chạy API

```bash
python app.py
```

API sẽ chạy tại: `http://localhost:5000`

## 🌐 Deploy lên Render (Free)

### Cách 1: Deploy từ GitHub (Recommend)

1. **Tạo GitHub Repository**

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/football-predictor.git
git push -u origin main
```

2. **Deploy trên Render**

- Vào [render.com](https://render.com) → Sign up (miễn phí)
- Click **"New"** → **"Web Service"**
- Connect GitHub repository
- Settings:
  - **Name**: `football-predictor`
  - **Environment**: `Python 3`
  - **Build Command**: `pip install -r requirements.txt`
  - **Start Command**: `gunicorn app:app`
- Click **"Create Web Service"**

3. **Đợi deploy** (3-5 phút)

API sẽ online tại: `https://football-predictor.onrender.com`

### Cách 2: Deploy nếu model quá lớn (>100MB)

Nếu file `football_model_v2.joblib` > 100MB, làm theo cách này:

1. **Upload model lên Google Drive**
   - Upload file model lên Google Drive
   - Share link (Anyone with the link can view)
   - Copy File ID từ link: `https://drive.google.com/file/d/FILE_ID_HERE/view`

2. **Sửa app.py** (thêm vào đầu file):

```python
import requests
import os

MODEL_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"

if not os.path.exists('football_model_v2.joblib'):
    print("Downloading model from Google Drive...")
    response = requests.get(MODEL_URL)
    with open('football_model_v2.joblib', 'wb') as f:
        f.write(response.content)
    print("✓ Model downloaded")
```

3. **Thêm vào requirements.txt**:
```
requests==2.31.0
```

## 📡 API Endpoints

### 1. Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "features_count": 45
}
```

### 2. Get Required Features

```bash
GET /features
```

Trả về danh sách tất cả features cần thiết.

### 3. Simple Prediction (Recommend)

```bash
POST /predict-simple
Content-Type: application/json

{
  "home_team": {
    "name": "Manchester City",
    "avg_goals_scored": 2.1,
    "avg_goals_conceded": 1.0,
    "avg_xg": 1.9,
    "avg_xg_conceded": 1.2,
    "avg_shots": 15.0,
    "avg_shots_conceded": 10.0,
    "avg_ppda": 8.5,
    "avg_deep": 6.0,
    "recent_form": [2, 2, 1, 2, 2],
    "clean_sheets_rate": 0.4
  },
  "away_team": {
    "name": "Liverpool",
    "avg_goals_scored": 1.8,
    "avg_goals_conceded": 1.3,
    "avg_xg": 1.7,
    "avg_xg_conceded": 1.4,
    "avg_shots": 13.0,
    "avg_shots_conceded": 11.0,
    "avg_ppda": 9.0,
    "avg_deep": 5.5,
    "recent_form": [1, 2, 0, 2, 1],
    "clean_sheets_rate": 0.3
  },
  "h2h_history": {
    "home_wins": 4,
    "draws": 2,
    "away_wins": 1
  }
}
```

**Response:**
```json
{
  "success": true,
  "match": {
    "home_team": "Manchester City",
    "away_team": "Liverpool"
  },
  "predictions": {
    "1x2": {
      "result": "Home Win",
      "probabilities": {
        "away_win": 0.15,
        "draw": 0.25,
        "home_win": 0.60
      }
    },
    "over_under": {
      "prediction": "Over 2.5",
      "probabilities": {
        "under": 0.35,
        "over": 0.65
      }
    },
    "score": {
      "home": 2.3,
      "away": 1.1,
      "total": 3.4
    }
  }
}
```

### 4. Full Prediction (Advanced)

```bash
POST /predict
Content-Type: application/json

{
  "features": {
    "h2h_home_wins": 0.5,
    "h2h_draws": 0.3,
    "h2h_away_wins": 0.2,
    // ... tất cả 45 features
  }
}
```

## 🔧 Input Fields Giải thích

### Team Stats

| Field | Mô tả | Ví dụ |
|-------|-------|-------|
| `avg_goals_scored` | Trung bình bàn thắng ghi được/trận | 2.1 |
| `avg_goals_conceded` | Trung bình bàn thắng thủng lưới/trận | 1.0 |
| `avg_xg` | Expected Goals trung bình | 1.9 |
| `avg_xg_conceded` | xG conceded trung bình | 1.2 |
| `avg_shots` | Số cú sút trung bình/trận | 15.0 |
| `avg_shots_conceded` | Số cú sút đối phương/trận | 10.0 |
| `avg_ppda` | PPDA (Passes Per Defensive Action) | 8.5 |
| `avg_deep` | Deep completions | 6.0 |
| `recent_form` | 5 trận gần nhất [0=thua, 1=hòa, 2=thắng] | [2,2,1,2,2] |
| `clean_sheets_rate` | Tỷ lệ giữ sạch lưới (0-1) | 0.4 |

### H2H History

| Field | Mô tả | Ví dụ |
|-------|-------|-------|
| `home_wins` | Số lần đội nhà thắng trong lịch sử đối đầu | 4 |
| `draws` | Số lần hòa | 2 |
| `away_wins` | Số lần đội khách thắng | 1 |

## 📱 Frontend Integration

### JavaScript Example

```javascript
async function predictMatch(homeTeam, awayTeam) {
  const response = await fetch('https://your-app.onrender.com/predict-simple', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      home_team: homeTeam,
      away_team: awayTeam,
      h2h_history: {
        home_wins: 3,
        draws: 2,
        away_wins: 1
      }
    })
  });
  
  const data = await response.json();
  
  if (data.success) {
    console.log('Prediction:', data.predictions);
    console.log('1x2:', data.predictions['1x2'].result);
    console.log('Score:', data.predictions.score);
  }
}
```

### Python Example

```python
import requests

url = "https://your-app.onrender.com/predict-simple"

payload = {
    "home_team": {
        "name": "Arsenal",
        "avg_goals_scored": 2.0,
        "avg_goals_conceded": 1.1,
        "avg_xg": 1.8,
        "avg_xg_conceded": 1.0,
        "avg_shots": 14.0,
        "avg_shots_conceded": 9.0,
        "avg_ppda": 9.0,
        "avg_deep": 5.5,
        "recent_form": [2, 2, 2, 1, 2],
        "clean_sheets_rate": 0.45
    },
    "away_team": {
        "name": "Chelsea",
        "avg_goals_scored": 1.6,
        "avg_goals_conceded": 1.4,
        "avg_xg": 1.5,
        "avg_xg_conceded": 1.3,
        "avg_shots": 12.0,
        "avg_shots_conceded": 11.0,
        "avg_ppda": 10.0,
        "avg_deep": 4.8,
        "recent_form": [1, 0, 2, 1, 0],
        "clean_sheets_rate": 0.25
    },
    "h2h_history": {
        "home_wins": 5,
        "draws": 3,
        "away_wins": 2
    }
}

response = requests.post(url, json=payload)
print(response.json())
```

## ⚠️ Lưu ý

### 1. Cold Start (Render Free Tier)
- API "ngủ" sau 15 phút không hoạt động
- Request đầu tiên có thể mất 30-60s để "đánh thức"
- Các request sau sẽ nhanh

### 2. Model Size
- Nếu model > 100MB → Upload lên Google Drive và download khi deploy
- Render free tier giới hạn ~500MB slug size

### 3. Rate Limiting
- Render free tier: 750 giờ/tháng
- Đủ cho development/demo

### 4. CORS
- API đã enable CORS → Có thể gọi từ bất kỳ domain nào
- Production: Nên giới hạn allowed origins

## 🔒 Security (Optional)

Thêm API Key authentication:

```python
# Thêm vào app.py
API_KEY = "your-secret-key-here"

@app.before_request
def check_api_key():
    if request.path not in ['/', '/health', '/features']:
        key = request.headers.get('X-API-Key')
        if key != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401
```

Frontend gọi:
```javascript
fetch(url, {
  headers: {
    'X-API-Key': 'your-secret-key-here',
    'Content-Type': 'application/json'
  }
})
```

## 📊 Model Performance

Model được train trên data từ 6 giải đấu:
- EPL (English Premier League)
- La Liga
- Serie A
- Bundesliga
- Ligue 1
- RFPL

Sử dụng ensemble của Random Forest và Gradient Boosting.

## 🆘 Troubleshooting

**Lỗi: "Model not found"**
- Đảm bảo file `football_model_v2.joblib` có trong repo
- Hoặc đã setup download từ Google Drive

**Lỗi: "Missing features"**
- Check response để xem features nào còn thiếu
- API sẽ tự động fill 0 cho missing features

**Deploy thất bại**
- Check Build Logs trên Render
- Đảm bảo `requirements.txt` đúng format
- Check Python version compatibility

## 📝 License

MIT License



