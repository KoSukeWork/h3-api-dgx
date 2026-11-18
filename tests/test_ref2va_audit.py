import json
import struct

import pytest

from scripts import audit_ref2va_adapter as audit


def test_summary_distinguishes_original_and_fused_ranks():
    header = {
        "__metadata__": {"alpha": "8"},
        "blocks.0.attn.to_q.lora_A.default.weight": {"dtype": "BF16", "shape": [128, 5376]},
        "blocks.0.attn.qkv_proj.lora_A.weight": {"dtype": "BF16", "shape": [384, 5376]},
        "blocks.0.attn.qkv_proj.alpha": {"dtype": "F32", "shape": []},
    }
    summary = audit.summarize(header)
    assert summary["tensor_count"] == 3
    assert summary["a_rank_counts"] == {"128": 1, "384": 1}
    assert summary["dtype_counts"] == {"BF16": 2, "F32": 1}


def test_header_audit_is_bounded_and_does_not_claim_full_sha(monkeypatch):
    header = {"w": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
    encoded = json.dumps(header).encode()
    raw = struct.pack("<Q", len(encoded)) + encoded
    monkeypatch.setitem(audit.FILES, "test", {"file": "fixture.safetensors", "bytes": len(raw) + 2})
    calls = []

    def read(url, start, size):
        calls.append((start, size))
        assert start + size <= len(raw)  # No tensor payload requested.
        return raw[start:start + size]

    monkeypatch.setattr(audit, "remote_range", read)
    result = audit.audit("test")
    assert calls == [(0, 8), (8, len(encoded))]
    assert result["full_file_sha256_verified"] is False
    monkeypatch.setitem(audit.FILES["test"], "bytes", 1)
    with pytest.raises(ValueError, match="offsets"):
        audit.audit("test")


def test_mapping_requires_reviewable_literal_and_complete_modules():
    with pytest.raises(ValueError, match="literal missing"):
        audit.audit_mapping({}, "pass")
    source = "param_names_mapping: dict = field(default_factory=lambda: {})"
    with pytest.raises(ValueError, match="Missing or unexpected"):
        audit.audit_mapping({}, source)
