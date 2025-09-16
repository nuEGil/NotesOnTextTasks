import requests
'''Predefined URL to connect to a local server thread. This listener is the client
All it does is constantly listen for text the user types. Once it gets a repsonse from 
the server, it will show up here. 
'''
URL = "http://127.0.0.1:8000/hash"

while True:
    text = input("Enter text (or 'quit' to exit): ").strip()
    if text.lower() == "quit":
        break

    resp = requests.post(URL, params={"text": text})
    if resp.status_code == 200:
        data = resp.json()
        print(f"Text: {data['text']}")
        print(f"Hash: {data['hash']}\n")
    else:
        print("Error:", resp.status_code, resp.text)
