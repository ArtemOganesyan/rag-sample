#!/bin/bash
cd /home/ubuntu/rag-sample/hermes-llama
source venv/bin/activate
nohup uvicorn hermes_llama:app --host 0.0.0.0 --port 8001 --workers 1 > model_server.log 2>&1 &
deactivate