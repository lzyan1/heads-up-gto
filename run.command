#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python run.py --open
else
  exec python3 run.py --open
fi
