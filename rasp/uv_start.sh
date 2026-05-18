#!/bin/bash

screen -dmS wateringcat_uv uv run uvicorn src.main:app --host 0.0.0.0 --port 8000