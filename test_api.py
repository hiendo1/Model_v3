"""
Test script cho Football Prediction API
Chạy file này để test API sau khi đã start server
"""

import requests
import json

# Thay đổi URL này khi deploy lên Render
API_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing /health endpoint ===")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_features():
    """Test features endpoint"""
    print("\n=== Testing /features endpoint ===")
    try:
        response = requests.get(f"{API_URL}/features")
        data = response.json()
        print(f"Status: {response.status_code}")
        print(f"Total features: {data['total_features']}")
        print(f"Feature groups: {list(data['groups'].keys())}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_simple():
    """Test predict-simple endpoint với data mẫu"""
    print("\n=== Testing /predict-simple endpoint ===")
    
    payload = {
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
    
    try:
        print(f"Request payload:")
        print(json.dumps(payload, indent=2))
        
        response = requests.post(
            f"{API_URL}/predict-simple",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Prediction successful!")
            print(f"\nMatch: {data['match']['home_team']} vs {data['match']['away_team']}")
            
            # 1x2 Result
            print(f"\n--- 1x2 Prediction ---")
            print(f"Result: {data['predictions']['1x2']['result']}")
            probs = data['predictions']['1x2']['probabilities']
            print(f"Home Win: {probs['home_win']:.1%}")
            print(f"Draw: {probs['draw']:.1%}")
            print(f"Away Win: {probs['away_win']:.1%}")
            
            # Score
            print(f"\n--- Score Prediction ---")
            score = data['predictions']['score']
            print(f"Predicted Score: {score['home']} - {score['away']}")
            print(f"Total Goals: {score['total']}")
            
            # Over/Under
            print(f"\n--- Over/Under 2.5 ---")
            ou = data['predictions']['over_under']
            print(f"Prediction: {ou['prediction']}")
            print(f"Over: {ou['probabilities']['over']:.1%}")
            print(f"Under: {ou['probabilities']['under']:.1%}")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_simple_weak_team():
    """Test với đội yếu hơn"""
    print("\n=== Testing weak team scenario ===")
    
    payload = {
        "home_team": {
            "name": "Watford",
            "avg_goals_scored": 1.0,
            "avg_goals_conceded": 2.2,
            "avg_xg": 0.9,
            "avg_xg_conceded": 2.0,
            "avg_shots": 9.0,
            "avg_shots_conceded": 15.0,
            "avg_ppda": 12.0,
            "avg_deep": 3.0,
            "recent_form": [0, 0, 1, 0, 0],
            "clean_sheets_rate": 0.1
        },
        "away_team": {
            "name": "Manchester City",
            "avg_goals_scored": 2.5,
            "avg_goals_conceded": 0.8,
            "avg_xg": 2.3,
            "avg_xg_conceded": 0.9,
            "avg_shots": 18.0,
            "avg_shots_conceded": 8.0,
            "avg_ppda": 7.0,
            "avg_deep": 7.5,
            "recent_form": [2, 2, 2, 2, 1],
            "clean_sheets_rate": 0.5
        },
        "h2h_history": {
            "home_wins": 0,
            "draws": 1,
            "away_wins": 5
        }
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict-simple",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Match: {data['match']['home_team']} vs {data['match']['away_team']}")
            print(f"Result: {data['predictions']['1x2']['result']}")
            print(f"Score: {data['predictions']['score']['home']} - {data['predictions']['score']['away']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("=" * 60)
    print("FOOTBALL PREDICTION API - TEST SUITE")
    print("=" * 60)
    print(f"Testing API at: {API_URL}")
    
    results = {
        "Health Check": test_health(),
        "Features Endpoint": test_features(),
        "Simple Prediction": test_predict_simple(),
        "Weak Team Scenario": test_predict_simple_weak_team()
    }
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
