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


#### Acknowledgements

We reuse most of components in [Marius](https://github.com/marius-team/marius) because they are well-developed.
