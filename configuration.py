"""API key configuration.

Set the environment variable OPENAI_API_KEY before running rouge.py::

    export OPENAI_API_KEY="sk-..."

This file must NOT contain a real key.  It is intentionally excluded from
version control via .gitignore.
"""

import os


class Config:
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
