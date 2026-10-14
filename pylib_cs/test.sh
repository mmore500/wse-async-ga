#!/usr/bin/bash

set -e

cd "$(dirname "$0")" || exit 1

python3.11 -m pytest test/
python3.11 -m black --check -l 80 .
python3.11 -m ruff check .
