import time, requests

BASE = "http://localhost:8000"
zip_path = "/home/abhiyaan-cu/Yash/MLOps-Project-ME22B214/data/test/ETs.zip"

with open(zip_path, "rb") as f:
    r = requests.post(f"{BASE}/upload", files={"file": ("images.zip", f, "application/zip")})
r.raise_for_status()
job_id = r.json()["job_id"]
print("job_id:", job_id)

while True:
    s = requests.get(f"{BASE}/status/{job_id}")
    s.raise_for_status()
    data = s.json()
    print(data["stage"], data["progress"], data["message"])
    if data["stage"] == "done":
        break
    if data["stage"] == "failed":
        raise RuntimeError(data.get("error", "job failed"))
    time.sleep(2)

ply = requests.get(f"{BASE}/download/{job_id}")
ply.raise_for_status()
with open("result.ply", "wb") as f:
    f.write(ply.content)
print("Saved result.ply")