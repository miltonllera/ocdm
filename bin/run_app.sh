#!/usr/bin/env bash

# OCDM Standalone Latent Visualization App Runner
# Usage:
#   bash bin/run_app.sh [PORT] [ADDRESS]
#
# Examples:
#   bash bin/run_app.sh 8501 127.0.0.1   # Local / SSH port forwarding
#   bash bin/run_app.sh 8501 0.0.0.0     # Direct network access

PORT=${1:-8501}
ADDRESS=${2:-"0.0.0.0"}

echo "=========================================================="
echo " Starting OCDM Latent Space Visualization App"
echo " Address: http://${ADDRESS}:${PORT}"
echo " For SSH port forwarding run on local machine:"
echo "   ssh -N -f -L ${PORT}:localhost:${PORT} user@remote-host"
echo "=========================================================="

uv run streamlit run scripts/app.py \
    --server.port "${PORT}" \
    --server.address "${ADDRESS}" \
    --server.headless true
