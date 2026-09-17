# Verified environment

The committed results were produced with the following environment. Training used CPU tensors even though the installed PyTorch build reported CUDA support.

| Item | Value |
| --- | --- |
| Operating system | Windows 11 |
| Python | 3.13.3 |
| PyTorch | 2.6.0+cu124 |
| Matplotlib | 3.10.1 |
| Execution device | CPU |
| PyTorch CPU threads | 16 |
| Random seed | 42 |

The 2,000-step baseline took 103.60 seconds on the verified machine. Wall-clock time will vary across CPUs; compare the loss traces and generated samples rather than expecting identical timing.
