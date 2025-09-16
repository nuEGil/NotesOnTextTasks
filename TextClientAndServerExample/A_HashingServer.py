from fastapi import FastAPI
import uvicorn
import hashlib

app = FastAPI()

@app.post("/hash")
def make_hash(text: str):
    # Generate a SHA256 hash of the input text
    h = hashlib.sha256(text.encode()).hexdigest()
    return {"text": text, "hash": h}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
