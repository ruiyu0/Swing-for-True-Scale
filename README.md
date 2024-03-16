# Swing for True Scale in Dual Camera Mode
The repository contains the code for "[Be Real in Scale: Swing for True Scale in Dual Camera Mode](https://jianwang-cmu.github.io/23realScale/Swing_for_True_Scale__ISMAR_2023_.pdf)" ISMAR 2023.

## Environment
The scale estimation code was developed using Python. You can use the following commands to set up the environment in Anaconda or Miniconda:
```
conda create -n swing4scale python=3.9
conda activate swing4scale
conda install pip
pip install open3d
pip install superpose3d
pip install mediapipe
```

## Face scale estimation
We provide sample data in the [data](https://github.com/ruiyu0/Swing-for-True-Scale/tree/main/data) folder collected by a calibrated Samsung Galaxy S22. To estimate the pupil distance (PD), simply run:

```
python main_pupil_dist_estimate.py
```


## Citation
If you use the code in your research, please cite the paper:
```
@inproceedings{yu2023real,
  title={Be Real in Scale: Swing for True Scale in Dual Camera Mode},
  author={Yu, Rui and Wang, Jian and Ma, Sizhuo and Huang, Sharon X and Krishnan, Gurunandan and Wu, Yicheng},
  booktitle={IEEE International Symposium on Mixed and Augmented Reality (ISMAR)},
  pages={1231--1239},
  year={2023}
}
```
