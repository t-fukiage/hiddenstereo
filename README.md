# hiddenstereo

These are python scripts that generate a Hidden Stereo pair.  
As for the details about the algorithm, please see:
<http://www.kecl.ntt.co.jp/human/hiddenstereo/>

## Dependency  

These scripts are written under Python 2.  
The libraries used in the scripts include:

* pyPyrTools  
<https://github.com/LabForComputationalVision/pyPyrTools>

* numpy

* opencv3

## gen_hiddenstereo.py  

This script generate a Hidden Stereo pair from a 2D image and a disparity map.
Please directly edit the script file to specify the file names of input images.

## stereo_transform.py  

This script transforms an arbitrary standard stereo pair into a Hidden Stereo pair.  
Please directly edit the script file to specify the file names of input images.

## Citation

Taiki Fukiage, Takahiro Kawabe, and Shin'ya Nishida
"Hiding of Phase-Based Stereo Disparity for Ghost-Free Viewing Without Glasses."
ACM Transaction on Graphics (Proc. of SIGGRAPH), 36(4):147, 2017.

@article{FukiageKN2017,
author = { Taiki Fukiage and
Takahiro Kawabe and
Shin'ya Nishida
},
title     = {Hiding of Phase-Based Stereo Disparity for Ghost-Free Viewing Without Glasses},
journal   = {ACM Transactions on Graphics (Proc. of SIGGRAPH 2017)},
year      = {2017},
volume    = {36},
number    = {4}
}
