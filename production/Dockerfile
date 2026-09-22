# Reuse the exact runtime that successfully generated the Turbo comparison.
FROM h3-bench:gb10-turbo
USER root
RUN test "$(uname -m)" = aarch64
RUN --mount=type=bind,source=benchmark/patch_lora_compat.py,target=/tmp/patch_lora_compat.py,ro \
    /usr/bin/python3.12 /tmp/patch_lora_compat.py
WORKDIR /opt/h3-api/app
COPY h3_api/ ./h3_api/
COPY examples/ ./examples/
USER h3:h3
CMD ["/opt/h3-api/app/.venv/bin/uvicorn", "h3_api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
