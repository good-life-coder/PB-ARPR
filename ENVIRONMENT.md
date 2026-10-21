# Environment

Observed interpreter: Python 3.10.21; package metadata is recorded in configs/current_environment.json. Historical Phase13B environment.txt records Python3.10.21, torch2.9.1+cu126, torchvision0.24.1+cu126, anomalib2.6.1, lightning2.6.6, CUDA12.6, Quadro RTX4000 8192MiB. Original executable/user paths are redacted in publication copies.

requirements.txt pins installed direct dependencies; it is not a complete resolver lock or proof that every version is publicly installable today. Obtain matching torch/torchvision CUDA wheels from [official instructions](https://pytorch.org/get-started/locally/), then install remaining requirements into a separate environment. Local GPU driver, CUDA wheel compatibility and platform-specific Windows helper imports must be checked. Some scripts use msvcrt/winreg/Windows resource accounting; Linux portability is UNVERIFIED.

No installation, dependency upgrade, model download or training was performed. Metadata inspection, compile/AST checks, aggregate arithmetic and small synthetic metric checks are the only executed checks. No full environment binary is bundled.
