#!/bin/bash

# Export the ip address
# PUBLIC_HOSTNAME=$(curl -s ifconfig.me)
export PUBLIC_HOSTNAME="localhost"

# Define and export port variables
export SHOPPING_PORT=7770
export SHOPPING_ADMIN_PORT=7780
export REDDIT_PORT=9999
export GITLAB_PORT=8023
export MAP_PORT=3000

# Define and export URL variables, these URLs are used for Docker initialization
export SHOPPING_URL="http://${PUBLIC_HOSTNAME}:${SHOPPING_PORT}"
export SHOPPING_ADMIN_URL="http://${PUBLIC_HOSTNAME}:${SHOPPING_ADMIN_PORT}/admin"
export REDDIT_URL="http://${PUBLIC_HOSTNAME}:${REDDIT_PORT}/forums/all"
export GITLAB_URL="http://${PUBLIC_HOSTNAME}:${GITLAB_PORT}/explore"
export MAP_URL="http://${PUBLIC_HOSTNAME}:${MAP_PORT}"

# These variables are the starting URL variables, use these variables for replacement
export SHOPPING="http://${PUBLIC_HOSTNAME}:${SHOPPING_PORT}"
export SHOPPING_ADMIN="http://${PUBLIC_HOSTNAME}:${SHOPPING_ADMIN_PORT}/admin"
export REDDIT="http://${PUBLIC_HOSTNAME}:${REDDIT_PORT}"
export GITLAB="http://${PUBLIC_HOSTNAME}:${GITLAB_PORT}"
export MAP="http://${PUBLIC_HOSTNAME}:${MAP_PORT}"

export ARCHIVES_LOCATION="/home/qianhuiwu/GUI-Libra/evaluation/WebArenaLiteV2/launcher/images"