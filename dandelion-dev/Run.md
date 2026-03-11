# GE2 build notes

The original `README.md` pins `CUDA 11.3 + torch 1.12.1`, but on this machine the working modern combination was:

- Python `3.10`
- PyTorch `2.9.0+cu128`
- CUDA toolkit `/usr/local/cuda-12.9`
- `g++ 11.4.0`
- CMake `4.1`

I verified that this combination can build `gege` successfully, and the built package imports correctly after a local install.

## Recommended env setup

```bash
conda create -n gege310 python=3.10 -y
conda activate gege310

python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0
python -m pip install numpy pandas omegaconf psutil GPUtil importlib_metadata

export CUDA_HOME=/usr/local/cuda-12.9
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH

# Change this if your GPU is not sm80.
export TORCH_CUDA_ARCH_LIST=8.0
```

## Build

Run from `/home/zwang269/code/GE2/dandelion-dev`:

```bash

cmake -S . -B build-cu129 \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build build-cu129 --target gege -j 4

```

## Install package

The repo's `pip-install` target now uses `--no-build-isolation`, which avoids the network dependency during install:

```bash
cmake --build build-cu129 --target pip-install -j 2
```

If your current Python environment is not writable, install from the generated package directory instead:

```bash
cd build-cu129/package
python -m pip install --user --no-build-isolation --no-deps .
```

If you used `--user`, make sure your local bin directory is on `PATH`:

```bash
export PATH=$HOME/.local/bin:$PATH
```

## Run

Run from `/home/zwang269/code/GE2/dandelion-dev` after the package is installed.

`export PATH=$HOME/.local/bin:/usr/local/cuda-12.9/bin:$PATH`
### 1. Preprocess a dataset

Example using the local Twitter SNAP edge list in this checkout:

```bash
gege_preprocess \
  --dataset custom \
  --edges raw/twitter_combined.txt \
  --output_directory datasets/twitter \
  --dataset_split 0.9 0.05 0.05 \
  --num_partitions 16 \
  --columns 0 1 \
  -d ' '
```

Example for a custom edge list:

```bash
gege_preprocess \
  --dataset custom \
  --edges /path/to/edges.tsv \
  --output_directory datasets/custom \
  --dataset_split 0.9 0.05 0.05 \
  --num_partitions 16
```
```bash
gege_preprocess \
  --dataset custom \
  --edges raw/soc-LiveJournal1.txt \
  --output_directory datasets/livejournal_16p \
  --dataset_split 0.9 0.05 0.05 \
  --num_partitions 16 \
  --columns 0 1 \
  -d ' '

```
The reader now treats `-d ' '` as generic whitespace and skips `#` comment lines, which matches the SNAP LiveJournal file format.


Livejournal sinlge GPU preprocess
```bash

gege_preprocess \
  --dataset custom \
  --edges raw/soc-LiveJournal1.txt \
  --output_directory datasets/livejournal \
  --dataset_split 0.9 0.05 0.05 \
  --num_partitions 1 \
  --columns 0 1 \
  -d $'\t'


```

### 2. Train

Example:

```bash
CUDA_VISIBLE_DEVICES=0 gege_train gege/configs/fb15k.yaml
```

If you want multi-GPU training, expose more than one device:

```bash
CUDA_VISIBLE_DEVICES=0,1 gege_train gege/configs/fb15k.yaml
```

### 3. Evaluate

Example:

```bash
CUDA_VISIBLE_DEVICES=0 gege_eval gege/configs/fb15k.yaml
```

### 4. Quick sanity check

After install, this should succeed:

```bash
python -c "import gege; print('gege import ok')"
```

### Notes

- The config files under `gege/configs/` assume CUDA devices by default.
- If your dataset path differs from the config defaults, update the YAML before training or evaluation.
- If your GPU architecture is not `sm80`, change `TORCH_CUDA_ARCH_LIST` before building.

## Verified working build

The following exact build worked in this workspace:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.0

cmake -S . -B build-cu129-final \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build build-cu129-final --target gege -j 4
```

