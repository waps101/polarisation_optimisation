# Shape-from-polarisation: a nonlinear least squares approach

This is a Matlab implementation of paper "Shape-from-polarisation: a nonlinear least squares approach".

Please cite the following paper if you use our code:

    @INPROCEEDINGS{yu2017shape, 
    author={Y. {Yu} and D. {Zhu} and W. A. P. {Smith}}, 
    booktitle={Proc. ICCV Workshop on Color and Photometry in Computer Vision}, 
    title={Shape-from-Polarisation: A Nonlinear Least Squares Approach}, 
    year={2017}, 
    pages={2969-2976}, 
    doi={10.1109/ICCVW.2017.350}, 
    }


## Test on synthetic data
```matlab
run demo_synthetic.m
```

## Synthetic data with additional noise
Change variable "noises" in file "demo_synthetic.m"


## Synthetic data with varying albedo
Change to "constantalbedo=false" in file "demo_synthetic.m"
