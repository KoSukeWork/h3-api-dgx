# Optional company gateway CA certificates

Place IT-approved **public CA certificates** in this directory on the DGX before
building. Use one PEM-encoded certificate per `.crt` file, e.g. `company-root.crt`.
Verify the SHA-256 fingerprint against a value supplied independently by your IT
administrator before adding a certificate. A valid CA format alone does not prove
that a certificate is trustworthy.

Do not add private keys, PFX/P12 files, or a website's leaf certificate. Do not
blindly copy certificates presented by a failed HTTPS connection. If IT supplies
a chain, ask for the root and necessary intermediate CA certificates separately.

The Dockerfile validates and imports the `.crt` files into the image's system
trust store. They are public certificates, not secrets, but the resulting image
will trust this company CA: keep the image internal. Changing or rotating a CA
requires rebuilding and recreating the containers; it is not a runtime bind mount.

The repository ignores local certificate files. The deployment package includes
only this README, never your actual certificates. On a network without a private
CA, leave this directory empty except for this README.
