# GE2 profiling notes

This file shows how to run full profiling for GE2 with:

- Nsight Systems (`nsys`) for end-to-end timeline profiling
- Nsight Compute (`ncu`) for kernel-level GPU profiling

These instructions assume:

- project root: `/home/zwang269/code/GE2/dandelion-dev`
- conda env: `gege310`
- CUDA toolkit: `/usr/local/cuda-12.9`
- installed package entrypoints: `gege_preprocess`, `gege_train`, `gege_eval`

## 1. Environment

```bash
cd /home/zwang269/code/GE2/dandelion-dev
conda activate gege310

export CUDA_HOME=/usr/local/cuda-12.9
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$HOME/.local/bin:$PATH
```

Profiler binaries on this machine:

- `nsys`
- `/usr/local/cuda-12.9/bin/ncu`

## 2. Preprocess and Training

#### preprocess first:
Only Preprocess once.
```bash
gege_preprocess   --dataset custom   --edges     /home/zwang269/code/GE2/dandelion-dev/raw/fb15k/train.txt     /home/zwang269/code/GE2/dandelion-dev/raw/fb15k/valid.txt     /home/zwang269/code/GE2/dandelion-dev/raw/fb15k/test.txt   --output_directory /home/zwang269/code/GE2/dandelion-dev/datasets/fb15k   --delim $'\t'   --num_partitions 1
```
#### Training
```bash
CUDA_VISIBLE_DEVICES=0 gege_train gege/configs/fb15k.yaml
```

If you want multi-GPU training, expose more than one device:

```bash
CUDA_VISIBLE_DEVICES=0,1 gege_train gege/configs/fb15k.yaml
```
## 3. Full timeline profiling with Nsight Systems

Use `nsys` first. It tells you whether time is spent in:

- CUDA kernels
- CUDA API calls
- CPU runtime
- Python / launcher overhead
- synchronization gaps

### Train profiling

```bash
mkdir -p profiles

CUDA_VISIBLE_DEVICES=0 nsys profile   --trace=cuda,nvtx,osrt   --cuda-event-trace=false   --sample=none   --cpuctxsw=none   --force-overwrite=true   --output profiles/trainer_profiling/single_GPU/baseline_result_new   gege_train profiles/trainer_profiling/single_GPU/fb15k.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=nvtx \
  --nvtx-capture=train_epoch_5@* \
  --capture-range-end=stop \
  --cuda-event-trace=false \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output profiles/gege_train_epoch5_nsys \
  gege_train gege/configs/fb15k.yaml
```
This produces:

- `profiles/gege_train_nsys.nsys-rep`
- possibly `profiles/gege_train_nsys.sqlite`

### Eval profiling

```bash
CUDA_VISIBLE_DEVICES=0 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output profiles/gege_eval_nsys \
  gege_eval gege/configs/fb15k.yaml
```

### Open the report

If you have the GUI:

```bash
nsys-ui profiles/gege_train_nsys.nsys-rep
```

Or export stats in terminal:

```bash
nsys stats profiles/gege_train_nsys.nsys-rep
```

## 4. Full kernel profiling with Nsight Compute

Use `ncu` after `nsys` identifies the expensive kernels.

`ncu` is much heavier than `nsys`, so expect slower runs.

### Train profiling
NCU
```bash
CUDA_VISIBLE_DEVICES=0 \
/usr/local/cuda-12.9/bin/ncu \
  --set full \
  --target-processes all \
  --force-overwrite \
  --export profiles/gege_train_ncu \
  gege_train gege/configs/fb15k.yaml
```

This produces a report such as:

- `profiles/gege_train_ncu.ncu-rep`

### Eval profiling

```bash
CUDA_VISIBLE_DEVICES=0 \
/usr/local/cuda-12.9/bin/ncu \
  --set full \
  --target-processes all \
  --force-overwrite \
  --export profiles/gege_eval_ncu \
  gege_eval gege/configs/fb15k.yaml
```

### Open the report

If you have the GUI:

```bash
ncu-ui profiles/gege_train_ncu.ncu-rep
```

Or inspect in terminal:

```bash
/usr/local/cuda-12.9/bin/ncu --import profiles/gege_train_ncu.ncu-rep
```

## 5. Lower-overhead targeted NCU run

If full `--set full` is too slow, start with a smaller metric set:

```bash
CUDA_VISIBLE_DEVICES=0 \
/usr/local/cuda-12.9/bin/ncu \
  --set launchstats \
  --target-processes all \
  --force-overwrite \
  --export profiles/gege_train_ncu_launchstats \
  gege_train gege/configs/fb15k.yaml
```

Then switch to `--set full` only after you know which kernels matter.

## 6. Profile the native binary instead of the Python entrypoint

If you want to remove Python launcher overhead from the trace, use the built executable directly:

```bash
CUDA_VISIBLE_DEVICES=0 \
./build-cu129-final/gege/gege_train gege/configs/fb15k.yaml
```

Equivalent `nsys` example:

```bash
CUDA_VISIBLE_DEVICES=0 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output profiles/gege_train_native_nsys \
  ./build-cu129-final/gege/gege_train gege/configs/fb15k.yaml
```

Equivalent `ncu` example:

```bash
CUDA_VISIBLE_DEVICES=0 \
/usr/local/cuda-12.9/bin/ncu \
  --set full \
  --target-processes all \
  --force-overwrite \
  --export profiles/gege_train_native_ncu \
  ./build-cu129-final/gege/gege_train gege/configs/fb15k.yaml
```

## 7. Recommended workflow

1. Run `nsys` on training.
2. Check whether the bottleneck is CUDA kernels, host-side stalls, or synchronization.
3. Pick the hottest kernel or phase.
4. Run `ncu` on that workload.
5. If profiling is too slow, reduce the dataset size or lower the number of epochs / steps in the YAML config.

## 8. Practical notes

- The configs under `gege/configs/` use CUDA by default.
- Use `CUDA_VISIBLE_DEVICES=0` first. Add more GPUs only if you specifically want multi-GPU profiling.
- `ncu --set full` can be very slow for long training jobs.
- For shorter profiling cycles, create a smaller config copy with fewer epochs or a smaller dataset.
- If you are profiling remote execution, save reports under `profiles/` and open them later in the GUI.
