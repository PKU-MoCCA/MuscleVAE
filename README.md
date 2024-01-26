# MuscleVAE: Model-Based Controllers of Muscle-Actuated Characters
An implemention of paper [MuscleVAE: Model-Based Controllers of Muscle-Actuated Characters](https://pku-mocca.github.io/MuscleVAE-page/).

[Yusen Feng](https://wangchek.github.io/),  [Xiyan Xu](https://xiyan-xu.github.io/), [Libin Liu](http://libliu.info/)

Project Page: https://pku-mocca.github.io/MuscleVAE-page/

## Installation

Install the packages listed in `requirements.txt`. This repo can be run on both `Linux` and `Windows`. But the render part is only supported on `Windows`. As of now, we only offer a basic OpenGL-based test render that can roughly display muscles and geometries, and it only supports the Windows operating system.

```
conda env create -n musclevae python=3.8
conda activate musclevae
pip install cython==0.29.30 tqdm scipy setuptools tensorboard tensorboardx psutil pyyaml cmake opt_einsum panda3d chardet trimesh
conda install mpi4py  
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Please to make sure the `cython` is the version `0.29.**` or `0.28.**`, otherwise you may encounter some errors.

Then install the package `ModifyODESrc`(Cpp) and `MuscleVAE`(Python) in this directory. You should  change into the folder of this project, and run
```
cd ModifyODESrc/
pip install -e .
cd ..
pip install -e .
```
`-e` means installing the package in "editable" mode. The package will change automatically without a reinstall after you make changes to the Python source files. But be careful and reinstall the package when you change the Cython files and type the `sh ./clear.sh` or `.\clear.cmd` and then reinstall when you change the Cpp files.


## Downloading the data
For the motion data, we use run, walk and jump motion from the LaFAN1 dataset. We offer the processed data in the following link. Please download it into `Data/ReferenceData/binary_data/`. 

```
https://drive.google.com/file/d/1b0iNcuDOpwi62Mu7Xn9j86obEvgs3QKb/view?usp=sharing
```

We also offer the `BVH` format in `Data/ReferenceData/walk_run_jump/`, you can use the following command to convert the `BVH` format to `binary` format. 

```
python ./Script/build_motion_dataset.py
``` 
It need an `.yml` file, such as `./Data/Parameters/muscle.yml`.

We offer a pretrained model and configuration file in the following link. Please download it into `Data/NNModel`.  
```
# low-level
https://drive.google.com/drive/folders/1Pd2zqtSflfgOYPJYUoq7DJPLfmzqsboS?usp=sharing
# high-level
https://drive.google.com/drive/folders/1UZZDs5-cvVrKEOf8NSMOmHmbKyrUIqAl?usp=sharing
```

## Playing

We offer three tasks now: `tracking` (low-level), `random generation` (low-level) and `velocity control` (high-level). All of them are in in the folder `PlayGround` and can be played by directly run the code. For more details, you can refer to the `PlayGround/random_generation.py`, `PlayGround/track_something.py` and `PlayGround/velocity_control.py`. As for the tracking task, we offer a code to visualize the world model in the position of reference character, which is in `PlayGround/visualize_world_model.py`. After typing the following command, you should select the configuration file `.yml` first, then the checkpoint `.data`.

```
python ./PlayGround/track_something.py --start_frame 10
python ./PlayGround/random_generation.py
python ./PlayGround/velocity_control.py
python ./PlayGround/visualize_world_model.py --start_frame 6000
```

## Training

Till now, we only offer the playing codes. The training codes will come soon.


## Citing

If you find our work useful in your research, please consider citing:

```
@inproceedings{feng2023musclevae,
    author = {Feng, Yusen and Xu, Xiyan and Liu, Libin},
    title = {MuscleVAE: Model-Based Controllers of Muscle-Actuated Characters},
    year = {2023},
    isbn = {9798400703157},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3610548.3618137},
    doi = {10.1145/3610548.3618137},
    booktitle = {SIGGRAPH Asia 2023 Conference Papers},
    articleno = {3},
    numpages = {11},
    series = {SA '23}
}
```

## Acknowledgement

The code is developed based open-source structure of [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE), and if you find world model is helpful for your research, please consider citing:

```
@article{Yao_2022,
   title={ControlVAE: Model-Based Learning of Generative Controllers for Physics-Based Characters},
   volume={41},
   ISSN={1557-7368},
   url={http://dx.doi.org/10.1145/3550454.3555434},
   DOI={10.1145/3550454.3555434},
   number={6},
   journal={ACM Transactions on Graphics},
   publisher={Association for Computing Machinery (ACM)},
   author={Yao, Heyuan and Song, Zhenhua and Chen, Baoquan and Liu, Libin},
   year={2022},
   month=nov, pages={1–16} }
```



