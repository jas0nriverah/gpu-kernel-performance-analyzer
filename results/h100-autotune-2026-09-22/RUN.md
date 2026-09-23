# H100 autotuning run

The original measured search returned exit status 0, passed correctness for all three block sizes, and selected block 256. That run independently recaptured only the Pareto/fastest finalist under its original protocol. Its discovery energy difference versus block 512 falls inside their summed MADs; the later all-speed-eligible follow-up and final uncertainty adjudication are recorded in [energy-review.md](energy-review.md).

| Block | Median runtime (ms/launch) | Median board energy (J/launch) | Runtime MAD (ms) | Energy MAD (J) |
|---:|---:|---:|---:|---:|
| 128 | 0.319687 | 0.14499995 | 0.0000228 | 0.0002249 |
| 256 | 0.288226 | 0.13757979 | 0.0000060 | 0.0005822 |
| 512 | 0.296508 | 0.13772788 | 0.0000265 | 0.0000609 |

The independent block-256 confirmation measured 0.288351 ms and 0.13698511 J per launch, within 20% of its initial medians and within 5% of the confirmed fastest candidate. Initial b256/b512 energies overlap under the summed-MAD screen, so the initial measurements alone do not establish a unique energy winner. The follow-up added an independent b512 capture; see [the final review](energy-review.md).

The exact command was:

```bash
/tmp/gpu-merge-review-py311/bin/python -m gpu_kernel_analyzer.cli autotune run \
  --binary /tmp/gpu-expansion-build/gpu_benchmark \
  --kernels vector_add --problem-sizes 67108864 --block-sizes 128 256 512 \
  --outdir results/h100-autotune-2026-09-22 --device 0 \
  --trials 3 --confirm-trials 3 --duration-seconds 15 --interval-seconds 0.2 \
  --idle-seconds 4 --trim-seconds 2 --timeout-seconds 120 \
  --warmups 10 --repeats 30 --seed 20260922 --speed-tolerance 0.05
```

The capture used NVIDIA H100 80GB HBM3, UUID `GPU-91e942dc-edbb-7f2a-9cea-167356bbd86a`, driver 595.71.05, 700 W power limit, 1980 MHz SM clock and 2619 MHz memory clock; MIG was disabled. Capture-time source commit was `6604687406afefae66c72ad8dad4199fc06ed37b` with a dirty working tree. Each capture manifest contains all source file hashes and the environment record. The benchmark binary SHA-256 is `fe41e52a9d9556642e513d791341e6f98e97d3d72f431f1e3c19b317ff04f5dc`; the autotuner source used for the capture was `29659b890a2c3d32537ac3841afcf249ab8fa69635b03deab8e975861a5b0bb5`. A subsequent tie-breaking-only change to the tuner was made after capture; no GPU result was rerun or altered.

Power is a board-level estimate from trapezoidal integration of `nvidia-smi` samples. It includes associated GPU circuitry; trial count and sensor averaging limit precision. See `report.json` and the nested manifests for raw samples, worker outputs, and hashes.
