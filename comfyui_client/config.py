"""Read local credentials for each execution; never serialize them in a workflow."""

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

CONFIG_PATH = Path(__file__).with_name("config.json")


def connection_settings(path=None):
    path = CONFIG_PATH if path is None else Path(path)
    try:
        if path.exists():
            config = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(config, dict):
                raise ValueError("Expected an object")
            # Never combine one source's URL with another source's secret.
            base, key = config.get("H3_API_URL"), config.get("H3_API_KEY")
        else:
            base, key = os.environ.get("H3_API_URL"), os.environ.get("H3_API_KEY")
    except (OSError, ValueError):
        raise ValueError("H3 config.json cannot be read; check JSON syntax and service-user permissions") from None
    if not isinstance(base, str) or not isinstance(key, str):
        raise ValueError(  # noqa: TRY004 - External configuration error, not a typed API argument.
            "Set both H3_API_URL and H3_API_KEY in adjacent config.json, or both in the process environment"
        )
    base, key = base.strip().rstrip("/"), key.strip()
    if (
        not key or key == "REPLACE_WITH_YOUR_API_KEY" or key.lower().startswith("bearer ")
        or not key.isascii() or any(ord(c) < 33 or ord(c) == 127 for c in key)
    ):
        raise ValueError("Set H3_API_KEY to the raw key without Bearer, whitespace or control characters")
    try:
        url = urlsplit(base)
        valid = (
            url.scheme in ("http", "https") and url.hostname
            and not url.username and not url.password and not url.query and not url.fragment
            and url.path in ("", "/")
            and not any(c.isspace() or ord(c) < 32 or c in '<>\\' for c in base)
        )
        _ = url.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("H3_API_URL must be an http(s) server address without /docs, credentials or query parameters")
    return base, key
