import os
import json
import sqlite3
import uvicorn
from pydantic import BaseModel 
from fastapi import FastAPI
from fastapi.responses import FileResponse

'''Example of hosting the database of books '''
app = FastAPI()

# Connect to the sql database 
database_dir = os.path.join(os.environ['BOOK_DIR'], "outputs/") 
database = os.path.join(database_dir, "textdata.db")
con = sqlite3.connect(database)
cur = con.cursor() # createa cursor object

class SearchRequest(BaseModel):
    text: str

# ---- serve the UI----
@app.get("/")
def root():
    return FileResponse("main.html")

@app.get("/style.css")
def css():
    return FileResponse("style.css")


@app.get("/script.js")
def js():
    return FileResponse("script.js")

# --- Search endpoint ---
@app.post("/search")
async def search(req: SearchRequest):
    # given a request of type Search Request, return the bottom result
    keyword = req.text.lower() # make it lower case  
    keyword = keyword.strip() # get rid of whitespace
    
    #so now we are getting all the text and wordsets from entries...
    cur.execute("SELECT text, wordset FROM entries")
    rows = cur.fetchall()
    matches = []
    # then from the list of rows that match we are trying to load json 
    # the wordset was a list [,,,] but it's treated as json
    # you should really vectorize the keywords, then hash those vectors. 
    # ok but after that, then  we do a comparison and see if the text is in the keyword set
    for entry_text, wordset_raw in rows:
        try:
            words = json.loads(wordset_raw)
        except:
            continue

        # Case-insensitive match
        if keyword in [w.lower() for w in words]:
            matches.append(entry_text)

    if not matches:
        return {"result": f"No matches found for {keyword}"}

    return {"result": "\n\n".join(matches)}

if __name__ == "__main__":
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
    con.close()  # close the sonnection to the database.
