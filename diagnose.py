import httpx
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8000"
API_KEY = os.environ.get("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

print(f"API_KEY loaded: {'yes' if API_KEY else 'NO - empty'}")
print(f"API_KEY value (first 4 chars): {API_KEY[:4] if API_KEY else 'none'}")
print()

response = httpx.get(
    f"{BASE_URL}/v1/autocomplete",
    params={"q": "diabetes", "limit": 7},
    headers=HEADERS,
    timeout=5.0,
)

print(f"Status code: {response.status_code}")
print(f"Response headers content-type: {response.headers.get('content-type')}")
print()
print("Raw response body:")
print(response.text)