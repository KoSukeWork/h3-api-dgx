#!/usr/bin/env bash
# Run as root on the DGX from this isolated checkout. No credentials in arguments.
set -euo pipefail
[[ $(uname -m) == aarch64 && $(uname -s) == Linux ]] || exit 2
production_container=${1:?provide the existing production container name}
[[ $production_container =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]+$ ]] || exit 2
: "${H3_REF2VA_CASE_DIR:?set private case directory}"
exec 9>/srv/h3/ref2va-test/production-pause.lock
flock -n 9 || exit 3
test_container=h3-ref2va-single-test
if docker container inspect "$test_container" >/dev/null 2>&1; then
    printf '%s\n' 'Test container name already exists; inspect it before retrying.'
    exit 4
fi
[[ $(docker inspect -f '{{.State.Running}}' "$production_container") == true ]] || exit 5
# Inspect inside the existing container; never print the key or entire env.
docker exec "$production_container" /opt/h3-api/app/.venv/bin/python -c '
import json, os, urllib.request
r = urllib.request.Request("http://127.0.0.1:8000/ready", headers={"Authorization": "Bearer " + os.environ["H3_API_KEY"]})
d = json.load(urllib.request.urlopen(r, timeout=10))
assert d.get("accepting_requests") is True, "Service not accepting requests"
assert d.get("active_job_id") is None, "Production task active"
jobs = d["jobs"]
assert isinstance(jobs, dict), "Unexpected queue schema"
assert all(not count or state in ("completed", "failed", "cancelled", "canceled") for state, count in jobs.items()), "Unfinished production jobs"
print(json.dumps({k: d.get(k) for k in ("backend_state", "active_job_id", "jobs")}))
'
paused=0
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if docker container inspect "$test_container" >/dev/null 2>&1; then
        docker stop --time 45 "$test_container" || true
        # Never restart production while the test is still running.
        if [[ $(docker inspect -f '{{.State.Running}}' "$test_container" 2>/dev/null || true) == true ]]; then
            printf '%s\n' 'Test shutdown failed; production remains stopped for GPU safety.'
            exit 6
        fi
    fi
    if [[ $paused == 1 ]]; then
        docker start "$production_container"
        printf '%s\n' 'Existing production container restarted (health must be rechecked).'
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Set before stop so the EXIT trap also covers interruption during shutdown.
paused=1
docker stop --time 60 "$production_container"
docker compose -f benchmark/ref2va/compose.yaml run --rm --no-deps --name "$test_container" test
