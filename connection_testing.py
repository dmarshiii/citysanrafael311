# Test the API
import requests

url = "https://services1.arcgis.com/YMIAeV29qfsu6BVo/arcgis/rest/services/Service_Requests_June_2022_Present/FeatureServer/0/query"

params = {
    'where': '1=1',
    'outFields': '*',
    'returnGeometry': 'true',
    'f': 'json',
    'resultRecordCount': 10
}

response = requests.get(url, params=params, timeout=30)
print(f"Status code: {response.status_code}")
print(f"Response length: {len(response.text)}")

data = response.json()
print(f"Keys in response: {data.keys()}")
print(f"Number of features: {len(data.get('features', []))}")

# Exit Python
exit()