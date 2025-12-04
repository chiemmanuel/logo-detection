import requests
from refresh_token import refresh_access_token
from datetime import datetime, timezone
import os


API_URL = "http://localhost:8080/api/tasks/"
PROJECT_ID = 1
API_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MTk3MDY3OSwiaWF0IjoxNzY0NzcwNjc5LCJqdGkiOiI2NmZkOTY1NmZiNzI0OTVlYTYyZjMwYjU5Y2JlZjQ4MyIsInVzZXJfaWQiOiIxIn0.3ldMbtSxd6jSPi6w52bD3RinGIOQVFIfjAzBS6VSQ3M"
local_folder = "datasets\\dataset_60"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

upload_images = 0
max_uploads = 500
start_time = datetime.now(timezone.utc)
print(f"Start time: {start_time.isoformat()}")


abs_local_folder = os.path.join("C:\\git_repos\\logo-detection", local_folder)

for root, dirs, files in os.walk(abs_local_folder):
    for file_name in files:
        if upload_images >= max_uploads:
            break
        if not (file_name.lower().endswith('.jpg') or file_name.lower().endswith('.png') or file_name.lower().endswith('.jpeg')):
            continue  # skip non-image files

        # Build relative path from DOCUMENT_ROOT folder to file
        abs_file_path = os.path.join(root, file_name)
        relative_file_path = os.path.relpath(abs_file_path, "C:\\git_repos\\logo-detection").replace("\\", "/")

        image_url = f"/data/local-files/?d={relative_file_path}"
        print(f"Uploading image: {image_url}")

        payload = {
            "project": PROJECT_ID,
            "data": {
                "image": image_url
            }
        }

        print("Refreshing access token...")
        API_TOKEN = refresh_access_token()
        headers["Authorization"] = f"Bearer {API_TOKEN}"

        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code in (200, 201):
            upload_images += 1
            print(f"✅ Uploaded: {relative_file_path}")
        else:
            print(f"❌ Failed: {relative_file_path}, Status: {response.status_code}, Message: {response.text}")

    if upload_images >= max_uploads:
        break

end_time = datetime.now(timezone.utc)
print(f"End time: {end_time.isoformat()}")
print(f"Total time taken: {end_time - start_time}")
