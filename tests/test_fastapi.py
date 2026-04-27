from fastapi.testclient import TestClient
from api.serve_app import fastapi_app

client = TestClient(fastapi_app)
try:
    res = client.post("/auth/token", json={"username": "admin", "password": "bad"})
    print("auth/token:", res.status_code)
except Exception as e:
    print("auth/token failed:", e)

try:
    res = client.get("/health")
    print("health:", res.status_code)
except Exception as e:
    print("health failed:", e)
