import requests
import json

# CORRECT data (should work)
correct_data = {
    "MedInc": 5.0,
    "HouseAge": 25.0,
    "AveRooms": 6.0,
    "AveBedrms": 1.2,
    "Population": 1200.0,
    "AveOccup": 3.0,
    "Latitude": 34.05,
    "Longitude": -118.25
}



print("Testing with CORRECT data:")
response = requests.post("http://127.0.0.1:5000/predict", json=correct_data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

print("Testing with INCORRECT data:")
response = requests.post("http://127.0.0.1:5000/predict", json=incorrect_data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")