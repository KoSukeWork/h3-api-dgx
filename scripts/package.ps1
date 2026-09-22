[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$h3Repo = Split-Path -Parent $PSScriptRoot
$h3Dist = Join-Path $h3Repo 'dist'
New-Item -ItemType Directory -Force -Path $h3Dist | Out-Null
$h3Archive = Join-Path $h3Dist 'h3-api-dgx.tar.gz'
$h3Directories = @('h3_api/', 'examples/', 'comfyui_client/', 'production/', 'bootstrap/',
    'scripts/', 'benchmark/', 'tests/', 'docs/')
$h3RootFiles = @('certs/README.md',
    'Dockerfile', '.dockerignore', '.env.example', '.gitignore', '.gitattributes',
    'pyproject.toml', 'uv.lock', 'inference-constraints.txt', 'predownload_h3_bf16.ps1',
    'README.md', 'AGENTS.md', 'LICENSE', 'NOTICE.md', 'SECURITY.md')
$h3Candidates = @(& git -c core.quotepath=false -C $h3Repo ls-files --cached --others --exclude-standard)
if ($LASTEXITCODE -ne 0) { throw 'Packaging requires a Git working directory so ignore rules can be respected.' }
$h3Items = @($h3Candidates | Sort-Object -Unique | Where-Object {
    $h3Name = $_
    ($h3RootFiles -contains $h3Name) -or
        ($h3Directories | Where-Object { $h3Name.StartsWith($_, [StringComparison]::Ordinal) })
})
foreach ($h3Name in $h3Items) {
    if ($h3Name -match '(^|/)comfyui_client/config\.json($|\.)') {
        throw "Refusing local client credentials: $h3Name"
    }
    if ($h3Name -match '(^|/)\.env($|\.(?!example$))|(^|/)__pycache__/|\.(pyc|crt|cer|der|pem|key|pfx|p12|safetensors|pt|pth|ckpt|mp4|wav|mp3|sqlite|sqlite3|db|log|whl|zip|tgz|7z|tar|tar\.gz)$') {
        throw "Refusing sensitive/generated archive entry: $h3Name"
    }
}
if ($h3Items.Count -eq 0) { throw 'No release files selected.' }
& tar -czf $h3Archive -C $h3Repo @h3Items
if ($LASTEXITCODE -ne 0) { throw "Packaging failed: $LASTEXITCODE" }
Get-FileHash -Algorithm SHA256 -LiteralPath $h3Archive
Write-Output "Standalone DGX package (no weights, secrets, certificates or Git history): $h3Archive"
