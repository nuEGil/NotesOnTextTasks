
'''Predefined URL to connect to a local server thread. This listener is the client
All it does is constantly listen for text the user types. Once it gets a repsonse from 
the server, it will show up here. 
'''
import requests

BASE_URL = "http://127.0.0.1:8000"

while True:
    text = input("Enter text (or 'quit'): ").strip()
    if text.lower() == "quit":
        break

    mode = input("Choose mode (hash/shift): ").strip().lower()
    endpoint = "/hash" if mode == "hash" else "/shift"

    resp = requests.post(BASE_URL + endpoint, params={"text": text})
    print(resp.json())