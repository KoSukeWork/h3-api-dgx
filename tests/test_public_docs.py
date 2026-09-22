import ipaddress
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def public_documents():
    return [
        ROOT / "README.md",
        ROOT / "NOTICE.md",
        ROOT / "SECURITY.md",
        ROOT / "AGENTS.md",
        *(ROOT / "docs").glob("*.md"),
        ROOT / "production/README.md",
        ROOT / "benchmark/README.md",
    ]


def test_docs_do_not_embed_personal_home_or_windows_paths():
    for path in public_documents():
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"/home/[A-Za-z0-9_.-]+", text), path
        assert not re.search(r"\b[A-Z]:[\\/]", text), path


def test_docs_do_not_embed_real_lan_addresses():
    for path in public_documents():
        text = path.read_text(encoding="utf-8")
        for literal in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
            try:
                address = ipaddress.ip_address(literal)
            except ValueError:
                continue  # Version strings with out-of-range octets are not addresses.
            assert address.is_loopback or not address.is_private, (path, literal)


def test_publication_notes_and_ignore_rules_are_shipped():
    package = (ROOT / "scripts/package.ps1").read_text(encoding="utf-8")
    assert "'SECURITY.md'" in package
    assert (ROOT / "docs/PUBLIC_RELEASE.md").is_file()
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    for pattern in (".env", "*.crt", "*.key", "*.pfx", "*.safetensors", "*.sqlite3", "*.log", "dist/"):
        assert pattern in ignored
