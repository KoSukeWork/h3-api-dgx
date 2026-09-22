"""DGX output acceptance: python scripts/check_output.py output.mp4

Decodes the whole file, checks stereo/24fps, and samples frames for black output.
This detects common failure modes, not semantic prompt adherence or latent NaNs.
"""

import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    info = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(args.video)]
        )
    )
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert Fraction(video["r_frame_rate"]) == 24, "Expected 24 fps"
    assert audio["channels"] == 2, "Expected stereo audio"
    assert int(audio["sample_rate"]) == 32000, "Expected 32 kHz audio"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(args.video),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    frames = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(args.video),
            "-vf",
            "fps=1,scale=32:32",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
    )
    assert frames, "No decoded frames"
    means = [sum(frames[i : i + 1024]) / len(frames[i : i + 1024]) for i in range(0, len(frames), 1024)]
    assert max(means) > 1, "All sampled frames are black; inspect the output"
    print(
        json.dumps(
            {
                "file": str(args.video),
                "width": video["width"],
                "height": video["height"],
                "fps": 24,
                "audio_channels": 2,
                "duration": info["format"]["duration"],
                "sample_frame_means": means,
                "decode": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
