# syntax=docker/dockerfile:1
# Registry manifest verified to include linux/arm64. Build on the GB10 host only.
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04@sha256:0230b7f243483cb15969fa3cc724a9459599604427052fc2a0d4291c7c0647dd
ARG TARGETARCH
RUN test "$TARGETARCH" = arm64 && test "$(uname -m)" = aarch64

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=12.1 \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
    MAX_JOBS=4 \
    CARGO_BUILD_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev build-essential git ffmpeg \
    pkg-config libssl-dev curl ca-certificates cmake ninja-build \
    protobuf-compiler libprotobuf-dev libnuma-dev libsndfile1 \
    && apt-get clean
RUN /usr/local/cuda/bin/nvcc --list-gpu-code | grep -x sm_121

# Fresh builds must trust the approved company CA BEFORE pip, rustup and git.
# Docker registry trust (and any HTTPS apt proxy) must also be configured on the host.
RUN --mount=type=bind,source=certs,target=/tmp/h3-certs,ro \
    --mount=type=bind,source=scripts/install_ca.sh,target=/tmp/install_ca.sh,ro \
    bash /tmp/install_ca.sh /tmp/h3-certs
ENV UV_NATIVE_TLS=true \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

RUN python3.12 -m venv /opt/h3-api/tools-venv \
    && /opt/h3-api/tools-venv/bin/pip install --no-cache-dir uv==0.10.6
ENV PATH=/opt/h3-api/tools-venv/bin:/opt/h3-api/inference-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV CARGO_HOME=/opt/h3-api/build-tools/cargo \
    RUSTUP_HOME=/opt/h3-api/build-tools/rustup \
    RUSTUP_TOOLCHAIN=1.92.0
RUN mkdir -p /opt/h3-api/build-tools \
    && curl --proto '=https' --tlsv1.2 --retry 3 -fsSL https://sh.rustup.rs \
        -o /opt/h3-api/build-tools/rustup-init.sh \
    && sh /opt/h3-api/build-tools/rustup-init.sh -y --no-modify-path \
        --profile minimal --default-toolchain 1.92.0
ENV PATH=/opt/h3-api/build-tools/cargo/bin:${PATH}

# Fixed native single-file loader; no quantization plugins/patches are enabled.
RUN git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang.git /opt/h3-api/sglang \
    && git -C /opt/h3-api/sglang fetch origin 70b5b03e78612c94f86ac98eb4d2d8d19ceda738 \
    && git -C /opt/h3-api/sglang checkout --detach 70b5b03e78612c94f86ac98eb4d2d8d19ceda738
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=inference-constraints.txt,target=/tmp/inference-constraints.txt,ro \
    python3.12 -m venv /opt/h3-api/inference-venv \
    && uv pip install --python /opt/h3-api/inference-venv/bin/python \
        --torch-backend=cu130 --prerelease=allow \
        --constraint /tmp/inference-constraints.txt -e '/opt/h3-api/sglang/python[diffusion]'
# Separate the install and audit layers so audit changes do not redo installation.
RUN --mount=type=bind,source=scripts/check_inference_env.py,target=/tmp/check_inference_env.py,ro \
    /opt/h3-api/inference-venv/bin/python /tmp/check_inference_env.py \
    && /opt/h3-api/inference-venv/bin/python -c \
        'import torch; import torch.backends.cusparselt as cs; print("torch", torch.__version__, "cuSPARSELt", cs.version()); assert cs.version() is not None' \
    && uv pip freeze --python /opt/h3-api/inference-venv/bin/python > /opt/h3-api/inference-resolved.txt

# Layout-only fix, deliberately AFTER dependency installation to preserve caches.
RUN --mount=type=bind,source=scripts/patch_h3_qkv.py,target=/tmp/patch_h3_qkv.py,ro \
    python3.12 /tmp/patch_h3_qkv.py

WORKDIR /opt/h3-api/app
COPY pyproject.toml uv.lock ./
COPY h3_api/ ./h3_api/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --python /usr/bin/python3.12
COPY examples/ ./examples/
COPY scripts/check_output.py ./scripts/check_output.py
RUN groupadd --gid 10001 h3 \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /var/lib/h3 h3
ENV XDG_CACHE_HOME=/srv/h3/data/cache \
    TORCH_HOME=/srv/h3/data/cache/torch \
    TRITON_CACHE_DIR=/srv/h3/data/cache/triton \
    CUDA_CACHE_PATH=/srv/h3/data/cache/cuda
USER h3:h3
EXPOSE 8000
CMD ["/opt/h3-api/app/.venv/bin/uvicorn", "h3_api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
