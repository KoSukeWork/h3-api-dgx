# Ref2VA isolated DGX smoke test — 2026-10-21

Result: **one reference-image generation completed on ARM64 GB10** using the
existing BF16 Ref2VA base and the pinned LightX2V original PEFT v0.1 adapter.
This validates this isolated loading/generation path, not visual quality,
the author's exact 544p recipe, or a production multi-profile API.

## Configuration and observations

- Source/runtime pins and adapter SHA256: [static audit](REF2VA_PREFLIGHT.md).
- Ref2VA base full SHA256 was verified before testing; no base-model redownload.
- Adapter downloaded on DGX, exact size and full SHA256 verified before loading.
- No quantization or dependency upgrades. Existing QKV/LoRA guards retained.
- Alpha 8, rank 128, runtime scale 1.0, dynamic application; **208 layers** logged active.
- 4 actual evaluations / 5 sigma points, video/audio shifts 12/3, guidance 1.
- Reference policy `lightx2v-match-v1`; original user prompt unchanged.
- Requested 480p, 16:9, 5 seconds; output **864×480, 124 frames, 24 fps**, about 5.17 seconds
  after native frame alignment; AAC stereo 32 kHz. Full ffmpeg decode succeeded.
- No explicit warmup. One cold generation only; not repeated steady-state timing.

| Measurement | Seconds |
| --- | ---: |
| Model startup to backend ready | 50.17 |
| Generation API wall time, excluding startup | 112.45 |
| Pipeline reported time | 110.96 |
| Text encoding | 4.33 |
| Reference visual encoding | 0.81 |
| Denoising | 73.50 |
| Decoding | 31.60 |

Startup + generation approximately 162.61 seconds, excluding asset verification,
container startup and production stop/restart. Do not compare this directly to
the historical 768p FL2VA/T2VA benchmark to claim a speedup.
Unified-memory peak was **not measured**; a Docker memory snapshot is not a GPU-memory peak.

## Issue found and corrected

The first loading attempt failed before generation: selecting a native Ref2VA
config alone leaves the pinned transformer's checkpoint validator defaulting to
FL2VA. Simply adding `--model-variant ref2va` to the old path would conflict with
`--model-subfolder .` and introduce double partition routing.

The isolated runner now creates a per-run symlink-only config view: native root
index/components for registry discovery, plus a `Ref2VA` link for semantic routing.
It explicitly supplies the matching model variant and subfolder. This preserves
the upstream checkpoint validator; the shared prepared configuration is unchanged.

## Operational boundary

The existing production queue was empty before each attempt. A wrapper stopped
only the named production container, ran the test alone, and restored that same
container afterward (including after the first failure). Post-test `/health` was
OK, `/ready` accepted requests with no active job, and no GPU compute process remained.
The production model was idle/unloaded after restart and will load on demand.

Private prompt/image/request/log/video remain outside Git in ignored local artifacts
and the isolated DGX data directory. No secrets or private samples are included here.
Local checks: 128 Python tests, 4 JS preview tests, scoped Ruff checks passed.

## Subsequent acceptance

The user inspected and accepted this isolated sample. The optional production
multi-profile upgrade was then implemented and passed Larry → Ref2VA → Larry
serial HTTP generation on GB10; see [production acceptance](PROFILES_DGX_ACCEPTANCE.md).
This document retains the earlier isolated-test measurements, not the later
production timings. Single-profile Larry deployments still reject Ref2VA.
