# Notes on web GUI
You can click the html file to get it going, but run the frontend like this
    python -m http.server 5500
    http://127.0.0.1:5500/main.html

Then run 
    python B_TextServer.py

## local hosting 
If you use The B_TextListener.py requests are posted with spaces represented by +. The web GUI will pass requests using the %20. You could run both the terminal text listener and the web app and the fastAPI server will listen for HTTP requests from both.

But, to enable that - had to add middleware with CORS allowing origins from specific ports for the local host. CORS = Cross Origin Resource Sharing. You could set it to '*' to allow requests to be made from any where, but then there some more work you should do on the security side. Not worth it. 

Now, For cloud platform, you can set security policies and do virtual private servers, so you could run the fastapi side or flask server from the vps on one end, then host the website on another and and have them talk back and forth. Websocket may be needed for some applicaitons. If its an image app, then you have to write some code to either have the user upload to the database - and have the server read the file path, or get something that can just pipe the pixels to the server itself. 

--> But I dont want that for this app. so switching to pyQT to get a native thing running. 
