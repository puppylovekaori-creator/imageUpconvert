param(
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$modelDir = Join-Path $repoRoot 'models\swinir'
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

$models = @(
    @{
        Name = '001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth'
        Url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth'
        Size = 67277475
    },
    @{
        Name = '001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
        Url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
        Size = 67869037
    },
    @{
        Name = '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth'
        Url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth'
        Size = 17147989
    },
    @{
        Name = '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth'
        Url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth'
        Size = 17225941
    },
    @{
        Name = '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
        Url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
        Size = 67129861
    },
    @{
        Name = '003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'
        Url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'
        Size = 142473939
    }
)

foreach ($model in $models) {
    $targetPath = Join-Path $modelDir $model.Name
    $tempPath = "$targetPath.part"

    if ((Test-Path $targetPath) -and -not $Force) {
        $existingLength = (Get-Item $targetPath).Length
        if ($existingLength -eq $model.Size) {
            Write-Host "skip $($model.Name)"
            continue
        }
        Write-Host "re-download $($model.Name) because size mismatch: $existingLength / expected $($model.Size)"
    }

    if (Test-Path $tempPath) {
        Remove-Item -LiteralPath $tempPath -Force
    }

    Write-Host "download $($model.Name)"
    Invoke-WebRequest -Uri $model.Url -OutFile $tempPath

    $downloadedLength = (Get-Item $tempPath).Length
    if ($downloadedLength -ne $model.Size) {
        Remove-Item -LiteralPath $tempPath -Force
        throw "Downloaded size mismatch for $($model.Name): $downloadedLength / expected $($model.Size)"
    }

    Move-Item -LiteralPath $tempPath -Destination $targetPath -Force
}

Write-Host 'model download completed'
