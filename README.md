# hiddenstereo

These are python scripts that generate a Hidden Stereo pair.  
As for the details about the algorithm, see:  
<http://www.kecl.ntt.co.jp/human/hiddenstereo/>

## Dependency  

These scripts are written under Python 2.  
The libraries used in the scripts include:

* pyPyrTools  
<https://github.com/LabForComputationalVision/pyPyrTools>

* numpy

* opencv3

## gen_hiddenstereo.py  

This script generate a Hidden Stereo pair from a 2D image and its disparity map.  
Please directly edit the file names in the script file to specify input images.  

## stereo_transform.py  

This script transforms an arbitrary standard stereo pair into a Hidden Stereo pair.  
Please directly edit the file names in the script file to specify input images.
