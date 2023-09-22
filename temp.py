import requests
from PIL import Image
import io

resp = requests.get('http://127.0.0.1:8000/my-first-api?name=rajat')
print(resp.text)