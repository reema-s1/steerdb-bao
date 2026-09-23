"""Line-flushed logging, so progress is visible when output is redirected to a file."""

from __future__ import annotations


def log(msg: str) -> None:
    print(msg, flush=True)
