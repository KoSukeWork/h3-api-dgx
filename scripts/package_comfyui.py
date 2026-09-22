"""Create a standalone ComfyUI node zip from an explicit, secret-free file list."""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIENT_FILES = (
    "__init__.py", "config.py", "config.example.json", "requirements.txt", "README.md", ".gitignore",
    "web/h3_video.js",
)


def build_archive(root=ROOT):
    root = Path(root)
    sources = [(root / "comfyui_client" / name, name) for name in CLIENT_FILES]
    sources += [(root / name, name) for name in ("LICENSE", "NOTICE.md")]
    for source, _ in sources:
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Expected a regular release source: {source.name}")
    directory = root / "dist"
    directory.mkdir(exist_ok=True)
    archive = directory / "h3-comfyui-client.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for source, name in sources:
            stream.write(source, "h3_api_client/" + name)
    return archive


if __name__ == "__main__":
    print(build_archive())
