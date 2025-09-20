#!/bin/bash
# Start the server in background
python B_HashingServer.py > server.log 2>&1 &

# Save PID if you want to kill later
SERVER_PID=$!

# Start the client in foreground
python B_TextListener.py

# When client exits, kill server
kill $SERVER_PID