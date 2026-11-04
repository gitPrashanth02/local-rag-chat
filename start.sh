#!/usr/bin/env bash
# Starts the backend and the chat page inside the container.
set -e

# 1. FastAPI backend, in the background (&). It listens only inside the container (127.0.0.1:8000);
#    visitors never reach it directly, the chat page calls it.
PORT=8000 python src/api.py &

# 2. Streamlit chat page on port 7860, the port Hugging Face Spaces shows to visitors.
#    The page says "API not reachable" for about a minute while the backend loads its models.
#    XSRF protection is off because it blocks file uploads inside the Spaces page frame.
exec streamlit run src/streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableXsrfProtection false
