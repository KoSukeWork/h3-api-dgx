"""Install an already downloaded pinned adapter, retaining the source and old files."""

import argparse
import os
import platform
import shutil
import tempfile
from pathlib import Path

from h3_api.ref2va import FILENAME, verify_lora


def install(source, destination):
    verify_lora(source)
    if destination.exists():
        verify_lora(destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".ref2va-", delete=False) as stream:
            temporary = Path(stream.name)
            with source.open("rb") as origin:
                shutil.copyfileobj(origin, stream, 16 * 1024 * 1024)
            stream.flush()
            os.fsync(stream.fileno())
        verify_lora(temporary)
        temporary.chmod(0o644)
        # Hard-link publication is atomic and refuses a destination created in the meantime.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        parser.error("Install on the ARM64 DGX only")
    destination = Path("/srv/h3/models/loras/lightx2v-ref2va-v01") / FILENAME
    install(args.source, destination)
    print(f"SHA-verified adapter installed: {destination}; source retained")
