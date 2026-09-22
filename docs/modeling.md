# Optional prediction and tuning

The ridge regression model predicts runtime and effective bandwidth from kernel name, problem size, and block size. It uses scikit-learn with the `[ml]` extra or a NumPy fallback. This is an exploratory extension to the measured benchmarking pipeline.

```bash
python -m gpu_kernel_analyzer model train \
  --run-dir outputs/run --model-out outputs/perf_model.json

python -m gpu_kernel_analyzer model predict \
  --model outputs/perf_model.json --kernel gemm_tiled --problem-size 512 --block-size 16

python -m gpu_kernel_analyzer model recommend-block-size \
  --model outputs/perf_model.json --kernel vector_add --problem-size 4194304 \
  --candidates 64,128,256,512

python -m gpu_kernel_analyzer advise --run-dir outputs/run
```

Predictions are tagged `predicted` and written separately from measured artifacts. Training supports repeated `--run-dir` arguments, but **do not pool different GPUs**: the current model has no device feature. Small datasets and unseen block sizes trigger warnings. Leave-one-out scores on nearby configurations do not establish generalization to a new workload or GPU; test on held-out sizes and independent runs before trusting recommendations.

The advisor is deterministic and offline. Its optional `--llm` switch is a stub, not a bundled AI integration. No API key is needed for ordinary use.
