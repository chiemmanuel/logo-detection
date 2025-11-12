import requests

def refresh_access_token():
    """
    Refresh the access token using the refresh token.
    """

    REFRESH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA2NDYwNzkxMywiaWF0IjoxNzU3NDA3OTEzLCJqdGkiOiI2MzFkYzc1NmFmNzg0YjY0OTdhYTE0YjgyNjhhYTZkNiIsInVzZXJfaWQiOiIxIn0.NcVDP2nq54xIQkXuk82HJlmBykJ1OE19csLmZV59xf0"
    
    LABEL_STUDIO_URL = "http://localhost:8080"

    response = requests.post(
        f"{LABEL_STUDIO_URL}/api/token/refresh/",
        headers={"Content-Type": "application/json"},
        json={"refresh": REFRESH_TOKEN}
    )

    if response.status_code == 200:
        access_token = response.json()["access"]
        return access_token
    else:
        print("❌ Failed to refresh token:", response.status_code, response.text)

refresh_access_token()

