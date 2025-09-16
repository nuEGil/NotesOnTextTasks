#!/bin/bash
# Start the server in background
python A_HashingServer.py > server.log 2>&1 &

# Save PID if you want to kill later
SERVER_PID=$!

# Start the client in foreground
python A_TextListener.py

# When client exits, kill server
kill $SERVER_PID