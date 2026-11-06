"""Launcher so the Injini agent can run `python injini.py <command>` from its home folder."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from injini_floor.cli import main  # noqa: E402

main()
