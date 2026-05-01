# SwinIR Models

This folder stores official SwinIR pretrained `.pth` files used by the GUI.

`setup.bat` downloads the GUI-supported official `2x / 4x` models here automatically.
If you need to fetch them again manually, run `download_models.bat`.

Recommended first model for low-invasive person-photo upscaling:

- `001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth`

Other useful official models:

- `001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`
- `002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth`
- `002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth`
- `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth`
- `003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth`

Official source:

- https://github.com/JingyunLiang/SwinIR/releases

This GUI detects the official model type from the filename. Keep the original filename.
