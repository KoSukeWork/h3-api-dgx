[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $PSScriptRoot 'preload\MiniMax-H3-BF16'),
    [switch]$IncludeRef2VA,
    [switch]$DryRun,
    [ValidateRange(1, 8)]
    [int]$MaxWorkers = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoId = 'Comfy-Org/MiniMax-H3'
$revision = '7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1'

$coreFiles = @(
    'diffusion_models/minimax_h3_fl2va_bf16.safetensors',
    'text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors',
    'vae/minimax_h3_video_vae_fp16.safetensors',
    'vae/minimax_h3_audio_vae_fp32.safetensors'
)

$ref2vaFiles = @(
    'diffusion_models/minimax_h3_ref2va_bf16.safetensors'
)

$files = @($coreFiles)
$estimatedDownloadGB = 123.6
if ($IncludeRef2VA) {
    $files += $ref2vaFiles
    $estimatedDownloadGB += 66.3
}

$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$modelsPath = Join-Path $destinationPath 'models'
$cachePath = Join-Path $destinationPath '.hf-cache'

Write-Host ''
Write-Host 'MiniMax H3 BF16 pre-download' -ForegroundColor Cyan
Write-Host "  Source:      $repoId"
Write-Host "  Revision:    $revision"
Write-Host "  Destination: $modelsPath"
Write-Host "  Cache:       $cachePath"
Write-Host ("  Selection:   FL2VA{0}" -f $(if ($IncludeRef2VA) { ' + Ref2VA' } else { '' }))
Write-Host ("  Estimated:   {0:N1} GB" -f $estimatedDownloadGB)
Write-Host '  Precision:   BF16 transformer/text encoder; native FP16/FP32 VAEs'
Write-Host ''

try {
    $driveRoot = [System.IO.Path]::GetPathRoot($destinationPath)
    $driveInfo = [System.IO.DriveInfo]::new($driveRoot)
    $freeGB = $driveInfo.AvailableFreeSpace / 1GB
    $recommendedFreeGB = $estimatedDownloadGB + 20
    Write-Host ("Free space on {0}: {1:N1} GiB" -f $driveRoot, $freeGB)
    if (-not $DryRun -and $freeGB -lt $recommendedFreeGB) {
        throw ("Not enough free space. Keep at least {0:N1} GiB free for the download and temporary metadata." -f $recommendedFreeGB)
    }
}
catch [System.ArgumentException] {
    Write-Warning 'Could not determine free disk space for the destination. Continuing.'
}

$hf = Get-Command 'hf' -ErrorAction SilentlyContinue
if (-not $hf) {
    throw @'
The Hugging Face CLI command `hf` was not found.
Install it first with:
  pip install -U huggingface_hub
'@
}

New-Item -ItemType Directory -Force -Path $modelsPath | Out-Null
New-Item -ItemType Directory -Force -Path $cachePath | Out-Null

# Keep all Hugging Face caches beside the downloaded models instead of using C:.
$env:HF_HOME = $cachePath
$env:HF_HUB_CACHE = Join-Path $cachePath 'hub'
$env:HF_XET_CACHE = Join-Path $cachePath 'xet'

$arguments = @('download', $repoId)
$arguments += $files
$arguments += @(
    '--revision', $revision,
    '--local-dir', $modelsPath,
    '--max-workers', $MaxWorkers.ToString()
)
if ($DryRun) {
    $arguments += '--dry-run'
}

Write-Host ''
if ($DryRun) {
    Write-Host 'Checking remote files; nothing will be downloaded...' -ForegroundColor Yellow
}
else {
    Write-Host 'Downloading. Re-run the same command after an interruption to resume.' -ForegroundColor Green
}
Write-Host ''

& $hf.Source @arguments
if ($LASTEXITCODE -ne 0) {
    throw "hf download failed with exit code $LASTEXITCODE"
}

if ($DryRun) {
    Write-Host ''
    Write-Host 'Dry run completed successfully.' -ForegroundColor Green
    exit 0
}

Write-Host ''
Write-Host 'Downloaded files:' -ForegroundColor Cyan
$totalBytes = 0L
foreach ($relativePath in $files) {
    $localPath = Join-Path $modelsPath ($relativePath -replace '/', [System.IO.Path]::DirectorySeparatorChar)
    if (-not (Test-Path -LiteralPath $localPath -PathType Leaf)) {
        throw "Expected file is missing after download: $localPath"
    }

    $item = Get-Item -LiteralPath $localPath
    $totalBytes += $item.Length
    Write-Host ("  {0,8:N2} GiB  {1}" -f ($item.Length / 1GB), $relativePath)
}

Write-Host ''
Write-Host ("Complete: {0:N2} GiB in {1}" -f ($totalBytes / 1GB), $modelsPath) -ForegroundColor Green
Write-Host 'The .hf-cache directory is only needed for resume metadata; do not copy it to the DGX.'
