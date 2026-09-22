import json
import struct

import pytest

from h3_api.prepare import inspect_header


def checkpoint(path, dtype="BF16", data=b"\x00\x00", offset=2):
    header = json.dumps({"weight": {"dtype": dtype, "shape": [1], "data_offsets": [0, offset]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def test_valid_bf16_header(tmp_path):
    path = tmp_path / "weight.safetensors"
    checkpoint(path)
    assert inspect_header(path, {"BF16"}) == {"tensor_count": 1, "dtypes": ["BF16"]}


def test_truncated_and_quantized_headers_rejected(tmp_path):
    path = tmp_path / "weight.safetensors"
    checkpoint(path, data=b"\x00")
    with pytest.raises(ValueError, match="Incomplete"):
        inspect_header(path, {"BF16"})
    checkpoint(path, dtype="I8", data=b"\x00", offset=1)
    with pytest.raises(ValueError, match="precision"):
        inspect_header(path, {"BF16"})
