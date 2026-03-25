#!/usr/bin/env bash

set -euo pipefail

if [[ -f ./env.sh ]]; then
    source ./env.sh
fi

# Get the directory where THIS script is located
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Python virtual environment location
MY_PY_ENV_DIR="$HOME/python-envs/z-image"

# Create / initialize the virtual environment if missing
if [[ ! -d "$MY_PY_ENV_DIR" ]]; then
    source "$SCRIPT_DIR/init.sh"
fi

# Ensure virtualenv activation script exists
if [[ ! -r "$MY_PY_ENV_DIR/bin/activate" ]]; then
    echo "Error: virtualenv activation script not found or not readable: $MY_PY_ENV_DIR" >&2
    exit 1
fi

# Activate the virtual environment
# shellcheck source=/dev/null
source "$MY_PY_ENV_DIR/bin/activate"

# Run the python script using the absolute path
exec python3 "$SCRIPT_DIR/z-image-2.py" "$@"
