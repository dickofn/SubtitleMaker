#!/usr/bin/env bash
# Launch SubtitleMaker using its own virtual environment.
# Linux/macOS counterpart to SubtitleMaker.bat.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$APP_DIR/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "Virtual environment not found at:" >&2
    echo "  $VENV_PY" >&2
    echo >&2
    echo "Recreate it with:" >&2
    echo "  python3 -m venv \"$APP_DIR/.venv\"" >&2
    echo "  \"$VENV_PY\" -m pip install -r \"$APP_DIR/requirements.txt\"" >&2
    echo >&2
    echo "Tkinter is not installed by pip. On Debian/Ubuntu:" >&2
    echo "  sudo apt install python3-tk" >&2
    exit 1
fi

exec "$VENV_PY" "$APP_DIR/SubtitleMaker.py" "$@"
