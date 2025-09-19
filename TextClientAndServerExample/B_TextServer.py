from fastapi import FastAPI, Query
import uvicorn
import hashlib
import urllib.parse

'''This is an example of using a state machine to get one end point, but 4 different functions for that end point 
Need the middleware to get stuff going in the web app. 
'''
app = FastAPI()

# --- Transform functions ---
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def shift_text(text: str) -> str:
    return "".join(
        chr((ord(c) + 4) % 127) if 32 <= ord(c) <= 126 else c
        for c in text
    )

def reverse_text(text: str) -> str:
    return text[::-1]

def upper_text(text: str) -> str:
    return text.upper()

# --- State machine (dispatch table) ---
TRANSFORMS = {
    "hash": hash_text,
    "ascii_shift": shift_text,
    "reverse": reverse_text,
    "upper": upper_text,
}

# --- Endpoint ---
@app.post("/transform")
def transform(text: str = Query(...), mode: str = Query("hash")):
    func = TRANSFORMS.get(mode)
    if func:
        return {"mode": mode, "text": text, "result": func(text)}
    else:
        return {"error": f"Unsupported mode: {mode}. Valid modes: {list(TRANSFORMS.keys())}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
