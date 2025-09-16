from fastapi import FastAPI
import uvicorn
import hashlib

app = FastAPI()

# Endpoint 1: Hashing
@app.post("/hash")
def make_hash(text: str):
    h = hashlib.sha256(text.encode()).hexdigest()
    return {"text": text, "hash": h}

# Endpoint 2: ASCII shift
@app.post("/shift")
def shift_text(text: str):
    shifted = "".join(chr((ord(c) + 4) % 127) if 32 <= ord(c) <= 126 else c for c in text)
    return {"original": text, "shifted": shifted}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)