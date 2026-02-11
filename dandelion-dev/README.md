GE^2

### Environment with Conda

        $ conda install -c "nvidia/label/cuda-11.3.1" cuda-toolkit
        $ conda install -c conda-forge cudnn # if needed
        $ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

### Build and Install

        $ mkdir build
        $ cd build
        $ cmake ..
        $ make gege -j           # build only python bindings
        $ make pip-install -j    # install pip package 

### Run Commands

        $ gege_preprocess --dataset twitter --output_dir datasets/twitter -ds 0.9 0.05 0.05 --num_partition 16
        $ CUDA_VISIBLE_DEVICES=0,1 gege_train gege/configs/fb15k.yaml

### Profiling

1. Build with profiling flags:

        $ mkdir -p build
        $ cd build
        $ cmake .. -DGEGE_ENABLE_PROFILING=ON -DGEGE_PROFILING_BACKEND=perf
        $ make gege_train -j

2. Run profiler (`perf` by default, sets `GEGE_PROFILE_TIMING=1` automatically):

        $ cd ..
        $ ./gege/scripts/profile_gege.sh --method perf -- ./gege/build/gege_train gege/configs/fb15k.yaml

   Output is written to `gege_profiles/<method>_<timestamp>/` with reports such as:
   `perf.data`, `perf.report.txt`, and `perf.script.txt`.

3. Optional `gprof` flow:

        $ cd build
        $ cmake .. -DGEGE_ENABLE_PROFILING=ON -DGEGE_PROFILING_BACKEND=gprof
        $ make gege_train -j
        $ cd ..
        $ ./gege/scripts/profile_gege.sh --method gprof -- ./gege/build/gege_train gege/configs/fb15k.yaml

4. GPU profiling with `nsys` (timeline + CUDA API + kernels):

        $ ./gege/scripts/profile_gege.sh --method nsys -- ./gege/build/gege_train gege/configs/fb15k.yaml

   Generates `nsys_profile.*` and `nsys.stats.txt` in the output directory.
   If you have multiple installations, you can select a specific binary with `--nsys-bin /path/to/nsys`.

5. GPU kernel metrics with `ncu`:

        $ ./gege/scripts/profile_gege.sh --method ncu -- ./gege/build/gege_train gege/configs/fb15k.yaml

   Optional filter for a specific kernel:

        $ ./gege/scripts/profile_gege.sh --method ncu --ncu-kernel ".*scatter.*" -- ./gege/build/gege_train gege/configs/fb15k.yaml

   If needed, select a specific binary with `--ncu-bin /path/to/ncu`.

#### Acknowledgements

We reuse most of components in [Marius](https://github.com/marius-team/marius) because they are well-developed.
