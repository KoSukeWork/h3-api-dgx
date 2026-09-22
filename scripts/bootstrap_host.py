"""Prepare named host directories and .env only; never install drivers or start services."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from production.manage import DATA, MODELS, PACKAGE, host_guard, old_services_stopped, write_env


def main():
    host_guard()
    old_services_stopped()
    for path in (MODELS, Path("/srv/h3/bootstrap-cache"), DATA):
        if path.is_symlink():
            raise RuntimeError(f"Refusing symlink: {path}")
        if not path.exists():
            path.mkdir(parents=True, mode=0o755)
            os.chown(path, 10001, 10001)
    if DATA.stat().st_uid != 10001 or any(DATA.iterdir()):
        raise RuntimeError("service-data must be empty and owned by UID 10001; no existing data is replaced")
    write_env(PACKAGE / ".env")
    print("Directories and .env prepared. No driver installation, downloads, service starts or deletions.")


if __name__ == "__main__":
    main()
