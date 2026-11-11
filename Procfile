web: gunicorn fast:app -w 1 --timeout 120 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
