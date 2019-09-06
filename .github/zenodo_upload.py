import requests
from os.path import expanduser

ACCESS_TOKEN  # this is where secret would go

headers = {"Content-Type": "application/json"}
r = requests.get(
    "https://sandbox.zenodo.org/api/deposit/depositions",
    params={'access_token': "ACCESS_TOKEN"},
    json={},
    headers=headers,
)
assert r.status_code // 100 != 4, r.json()
deposition_id = r.json()['id']

print(r.json())
