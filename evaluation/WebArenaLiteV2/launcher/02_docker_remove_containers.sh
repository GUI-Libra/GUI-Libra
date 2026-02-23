#!/bin/bash

docker stop shopping_admin forum gitlab shopping openstreetmap-website-db-1 openstreetmap-website-web-1
docker rm shopping_admin forum gitlab shopping openstreetmap-website-db-1 openstreetmap-website-web-1

PID=$(netstat -tulpn 2>/dev/null | grep ":${HOMEPAGE_PORT}" | awk '{print $7}' | cut -d'/' -f1)

if [ -n "$PID" ]; then
    echo "Found process using port ${HOMEPAGE_PORT}, PID: $PID, terminating..."
    kill -9 $PID
    echo "Process terminated"
else
    echo "No process found using port ${HOMEPAGE_PORT}"
fi