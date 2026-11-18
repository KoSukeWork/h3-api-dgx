# Isolated LightX2V Ref2VA smoke test

This directory contains the isolated test runner, not the production API deployment.
Existing single-profile Larry FL2VA protection is unchanged.
See [static audit](../../docs/REF2VA_PREFLIGHT.md) for source pins and limitations.
One GB10 smoke test completed; see [execution record](../../docs/REF2VA_DGX_TEST.md).
The user subsequently accepted the sample, and the separate multi-profile production
upgrade passed serial switching; see [production acceptance](../../docs/PROFILES_DGX_ACCEPTANCE.md).

Only run on ARM64 DGX Spark GB10. The derived image reuses the existing runtime,
adds no Python/GPU dependencies, and preserves the QKV and LoRA compatibility patches.
The two reference-geometry source files are checked by SHA256 before patching;
unknown or already modified sources fail closed. The opt-in `lightx2v-match-v1`
policy computes geometry during prequeue admission; both Qwen and VAE then use
the same cached dimensions. Without the environment flag, upstream 2048 behavior stays unchanged.
The pinned loader also requires explicit `--model-variant ref2va` for checkpoint
validation. Each isolated run creates a symlink-only config view with a native
root index for registry discovery and a `Ref2VA` subfolder for semantic routing.
Both point to the same prepared Ref2VA configuration; shared model files are not changed.

The original PEFT v0.1 adapter is downloaded at a fixed revision and verified in full.
Rank 128, alpha 8, scale 1.0, dynamic application, exactly 208 native modules expected.
No quantization. One cold sample, 4 evaluations (5 sigma points), shifts 12/3,
480p/16:9/5 seconds. This is **not** the author's 544p training-resolution reproduction.
No warmup or baseline speedup claims. Media checks cannot establish motion quality.

Prepare a **private, untracked** directory containing `reference.png` and `case.json`:

```json
{
  "image": "reference.png",
  "image_sha256": "<SHA256 of your PNG>",
  "seed": 42,
  "prompt_original": "Your original prompt",
  "prompt_reference_bound": "Use <Picture 1> as the reference. Your original prompt"
}
```

The container UID 10001 must be able to read this directory. Do not publish the
private case, request logs, SSH credentials, CA or API secrets. No network ports
are published by the test Compose project. `/srv/h3/models` is read-only;
all new output/cache/adapter files go under `/srv/h3/ref2va-test`.

On the DGX, create the isolated data directory (only if it does not already exist),
owned by UID/GID 10001, then set `H3_REF2VA_CASE_DIR` to your private case directory.
Optionally tag the inspected existing image ID with a dedicated local tag and set
`H3_REF2VA_BASE_IMAGE` to that tag. Dockerfile `FROM` does not accept a bare local image ID.

```bash
sudo --preserve-env=H3_REF2VA_CASE_DIR,H3_REF2VA_BASE_IMAGE docker compose -f benchmark/ref2va/compose.yaml build test
sudo --preserve-env=H3_REF2VA_CASE_DIR docker compose -f benchmark/ref2va/compose.yaml run --rm --no-deps download
```

Downloading/building does not need the GPU or stopping production. Before testing,
coordinate a maintenance window so clients do not submit new requests. The wrapper
refuses unfinished production jobs, temporarily stops the named existing container,
runs one test, and restarts that same production container on success or failure.
It refuses another GPU compute process and never kills unrelated GPU jobs.

```bash
sudo --preserve-env=H3_REF2VA_CASE_DIR bash benchmark/ref2va/with_production_paused.sh <existing-production-container>
```

The wrapper must run from the isolated repository root. Recheck production `/health`
and authenticated `/ready` afterward. OS shutdown/SIGKILL cannot execute a shell trap;
in that case inspect test/GPU state before restarting production manually.

Results: `/srv/h3/ref2va-test/runs/<timestamp>/` (`summary.json`, request, logs, MP4).
Download them privately and inspect identity, hand action, box opening, idle return,
foot sliding and camera stability. Successful media validation alone is not acceptance.
Multi-profile API switching is a later step after this test passes.
