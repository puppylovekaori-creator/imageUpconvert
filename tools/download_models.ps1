param(
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$vendorRoot = Join-Path $repoRoot 'vendor\gimp_upscale'
$targetDir = Join-Path $vendorRoot 'resrgan'
$requiredModels = @(
    'UltraSharp-4x',
    'RealESRGAN_General_x4_v3',
    'realesrgan-x4plus',
    'realesrgan-x4plus-anime',
    'AnimeSharp-4x',
    'realesr-animevideov3-x4'
)

function Test-BackendReady {
    param([string]$DirPath)
    if (-not (Test-Path (Join-Path $DirPath 'realesrgan-ncnn-vulkan.exe'))) {
        return $false
    }
    $modelDir = Join-Path $DirPath 'models'
    if (-not (Test-Path $modelDir)) {
        return $false
    }
    foreach ($model in $requiredModels) {
        if (-not (Test-Path (Join-Path $modelDir "$model.param"))) {
            return $false
        }
        if (-not (Test-Path (Join-Path $modelDir "$model.bin"))) {
            return $false
        }
    }
    return $true
}

if ((-not $Force) -and (Test-BackendReady -DirPath $targetDir)) {
    Write-Host 'skip existing backend'
    exit 0
}

$headers = @{
    'User-Agent' = 'imageUpconvert-setup'
    'Accept' = 'application/vnd.github+json'
}

Write-Host 'fetch latest gimp_upscale release metadata'
$release = Invoke-RestMethod -Uri 'https://api.github.com/repos/Nenotriple/gimp_upscale/releases/latest' -Headers $headers
$asset = $release.assets | Where-Object { $_.name -eq 'gimp3_upscale.zip' } | Select-Object -First 1
if (-not $asset) {
    $asset = $release.assets | Where-Object { $_.name -like 'gimp*_upscale.zip' } | Select-Object -First 1
}
if (-not $asset) {
    throw 'gimp_upscale release asset was not found.'
}

$tempRoot = Join-Path $env:TEMP ("imageupconvert_" + [guid]::NewGuid().ToString('N'))
$zipPath = Join-Path $tempRoot $asset.name
$extractRoot = Join-Path $tempRoot 'extract'

New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
New-Item -ItemType Directory -Force -Path $vendorRoot | Out-Null

try {
    Write-Host "download $($asset.name)"
    Invoke-WebRequest -Uri $asset.browser_download_url -Headers $headers -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $extractRoot -Force

    $resrganDir = Get-ChildItem -Path $extractRoot -Recurse -Directory -ErrorAction Stop |
        Where-Object { $_.Name -eq 'resrgan' } |
        Select-Object -First 1
    if (-not $resrganDir) {
        throw 'resrgan directory was not found in the downloaded archive.'
    }

    $stagingDir = Join-Path $tempRoot 'staging\resrgan'
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $stagingDir) | Out-Null
    Copy-Item -Path $resrganDir.FullName -Destination $stagingDir -Recurse -Force

    if (-not (Test-BackendReady -DirPath $stagingDir)) {
        throw 'downloaded backend is missing required executable or models.'
    }

    if (Test-Path $targetDir) {
        Remove-Item -LiteralPath $targetDir -Recurse -Force
    }
    Move-Item -LiteralPath $stagingDir -Destination $targetDir -Force

    $manifest = [ordered]@{
        tag_name = $release.tag_name
        release_name = $release.name
        asset_name = $asset.name
        asset_url = $asset.browser_download_url
        downloaded_at = (Get-Date).ToString('o')
    }
    $manifestPath = Join-Path $vendorRoot 'backend_manifest.json'
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $manifestPath -Encoding UTF8

    Write-Host 'downloaded backend models:'
    Get-ChildItem -Path (Join-Path $targetDir 'models') -Filter *.param |
        Select-Object -ExpandProperty BaseName |
        Sort-Object |
        ForEach-Object { Write-Host "  $_" }
}
finally {
    if (Test-Path $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}

Write-Host 'backend download completed'
