#!/usr/bin/env bash
# Runs inside the build container. Only public, IT-approved CA certificates belong here.
set -euo pipefail
cert_dir="${1:?Expected the directory of public CA certificates}"
shopt -s nullglob
certs=("$cert_dir"/*.crt)
if (( ${#certs[@]} == 0 )); then
    echo 'No custom CA certificates supplied; keeping Ubuntu system trust.'
    exit 0
fi
# Validate the entire input set before copying anything into the image trust store.
for cert in "${certs[@]}"; do
    name="${cert##*/}"
    if [[ -L "$cert" || ! -f "$cert" || ! "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*\.crt$ ]]; then
        echo "Invalid CA filename or file type: $name" >&2
        exit 1
    fi
    if grep -q 'PRIVATE KEY' "$cert"; then
        echo "Refusing a private key in $name; supply public CA certificates only." >&2
        exit 1
    fi
    count="$(grep -c -- '-----BEGIN CERTIFICATE-----' "$cert" || true)"
    if [[ "$count" != 1 ]]; then
        echo "$name must contain exactly one PEM certificate (not DER or a certificate bundle)." >&2
        exit 1
    fi
    openssl x509 -in "$cert" -noout -checkend 0
    constraints="$(openssl x509 -in "$cert" -noout -ext basicConstraints)"
    if [[ "$constraints" != *'CA:TRUE'* ]]; then
        echo "$name is not a CA certificate. Do not install a website leaf certificate." >&2
        exit 1
    fi
    # These public details allow the operator to audit the certificates being trusted.
    openssl x509 -in "$cert" -noout -subject -issuer -dates -fingerprint -sha256
done
install -d -m 755 /usr/local/share/ca-certificates/h3-company
for cert in "${certs[@]}"; do
    install -m 644 "$cert" "/usr/local/share/ca-certificates/h3-company/${cert##*/}"
done
update-ca-certificates
