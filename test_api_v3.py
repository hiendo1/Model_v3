import requests
import json

url = "api key"
data = {
    "home_name": "Arsenal FC",
    "away_name": "Liverpool FC"
}

print(f"Testing /predict-by-name with {data['home_name']} vs {data['away_name']}...")
response = requests.post(url, json=data)

if response.status_code == 200:
    print("Success!\n")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Error {response.status_code}: {response.text}")
