#!/bin/bash
cd /home/ubuntu/rag-sample/embedder
source venv/bin/activate
nohup uvicorn embedder_service:app --host 0.0.0.0 --port 8002 --workers 1 > model_server.log 2>&1 &
deactivate