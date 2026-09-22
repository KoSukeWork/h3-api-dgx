import asyncio
import json
from pathlib import Path


async def probe(path: Path) -> dict:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=60)
    except BaseException:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        raise
    if proc.returncode:
        raise ValueError(f"Invalid media: {err.decode(errors='replace')[:300]}")
    return json.loads(out)


async def validate_input(path: Path, kind: str):
    info = await probe(path)
    streams = info.get("streams", [])
    expected = "audio" if kind == "audio" else "video"
    if not any(s.get("codec_type") == expected for s in streams):
        raise ValueError(f"File has no {expected} stream")
    if kind != "image":
        duration = float(info.get("format", {}).get("duration", 0))
        if not 2 <= duration <= 15:
            raise ValueError("Reference audio/video must be between 2 and 15 seconds")


async def validate_output(path: Path):
    info = await probe(path)
    streams = info.get("streams", [])
    if not any(s.get("codec_type") == "video" for s in streams):
        raise ValueError("Backend output has no video stream")
    if not any(s.get("codec_type") == "audio" and s.get("channels") == 2 for s in streams):
        raise ValueError("Backend output has no stereo audio stream")
