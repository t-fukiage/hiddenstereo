# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:16:22 2016

@author: taikifukiage
"""


import cv2
import numpy as np

import copy
import time

#import pyPyrTools as ppt

from ppt_ext import SGpyr
from ppt_ext import SCFpyr_mod


#%%parameters

#input images

#cyclopean view image
rgb0 = cv2.imread('./imgs/center.png',1)

#disparity map
disp = cv2.imread('./imgs/disparity.png',0)


#Disparity range is rescaled using this value.(from -disp_range/2 to disp_range/2)
disp_range = 7.


#When this is True, only luminance channel is modulated
Luminance_mode = True

#make this variable "True" when you want to resize the input images
resize_img = False

#The input images are scaled in (w,h)=(imsize_x,imsize_y) when "resize_img = True"
imsize_x = 512
imsize_y = 512


#number of pyramid levels (including highest & lowest residuals)
sflevels = 7

#number of orientation bands
olevels = 8


#gamma correction (necessary for perfect cancellation of the disparity-inducer component if the monitor's gamma is not linearized)
HandleGamma = False
#gamma = 2.2

DR_low = 0
DR_high = 255

  
#%%

def srgb2rgb(img):
    
    out = np.zeros(img.shape)
    
    if False:
        #Adobe sRGB
        out[img<=0.0556]=img[img<=0.0556]/32.
        out[img>0.0556]=img[img>0.0556]**2.2
    else:
        out[img<= 0.040450]=img[img<= 0.040450]/12.92
        out[img> 0.040450]=((img[img> 0.040450] + 0.055)/1.055)**2.4
      
    return out

def rgb2srgb(img):
    
    out = np.zeros(img.shape)
    
    if False:
        #Adobe sRGB
        out[img<=0.00174]=img[img<=0.00174]*32.
        out[img>0.00174]=img[img>0.00174]**(1./2.2)
    else:
        out[img<=0.0031308]=img[img<=0.0031308]*12.92
        out[img>0.0031308]=(img[img>0.0031308]**(1./2.4))*1.055 - 0.055
      
    return out
    

#%%
if resize_img:
    rgb0 = cv2.resize(rgb0,(imsize_x,imsize_y))
    disp = cv2.resize(disp,(imsize_x,imsize_y))
    #rgb1 = cv2.resize(rgb1,(imsize_x,imsize_y))


#cv2.imwrite("orgL.png",rgb0)
#cv2.imwrite("orgR.png",rgb1)

#sumLR = rgb0/2. + rgb1/2.
#cv2.imwrite("orgL+R.png",sumLR)

    
if Luminance_mode:
    yuv0 = cv2.cvtColor(rgb0, cv2.COLOR_BGR2YUV)
    img0 = yuv0[:,:,0]
        
    #yuv1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2YUV)
    #img1 = yuv1[:,:,0]
    
    if HandleGamma:
        
        img0=srgb2rgb(np.float64(img0)/255.)*255.
        #img1=srgb2rgb(np.float64(img1)/255.)*255.
        
    original = img0.copy()
    

else:
    if HandleGamma:
        #rgb0 = ((rgb0/255.)**gamma)*255.
        rgb0=srgb2rgb(rgb0/255.)*255.
        #rgb1=srgb2rgb(rgb1/255.)*255.
           
    rgbimg0 = np.float64(rgb0)
    #rgbimg1 = np.float64(rgb1)
    original = rgbimg0.copy()
    

#%%
start = time.time()

print "decomposing images..."

if Luminance_mode:
    pyr0 = SCFpyr_mod(img0,sflevels-2,olevels-1)
    pyr0_2 = copy.deepcopy(pyr0)
    
    _pyr0 = copy.deepcopy(pyr0)
    
    #pyr1 = ppt.SCFpyr(img1,sflevels-2,olevels-1)#SCFpyr_ext(img1, sflevels-2, order=olevels-1, pyrType=pyrType)
else:
    pyrList0 = []
    #pyrList1 = []
    for i in xrange(3):
        pyr0 = SCFpyr_mod(rgbimg0[:,:,i],sflevels-2,olevels-1)
        #pyr1 = ppt.SCFpyr(rgbimg1[:,:,i],sflevels-2,olevels-1)
        pyrList0.append(pyr0)
        #pyrList1.append(pyr1)
    
    pyrList0_2 = copy.deepcopy(pyrList0)

    pyrList_shift = copy.deepcopy(pyrList0)

    


#%%
print "generating quadrature-phase components"

for s in range(pyr0.spyrHt()):
    for b in range(pyr0.numBands()):
        band_id = ((s*pyr0.numBands())+b)+1
        """
        ang0 = np.angle(_pyr0.pyr[band_id])
        ene = np.abs(_pyr0.pyr[band_id])
        if b < pyr0.numBands()/2:
            addcomp = -np.pi/2.
        else:
            addcomp = np.pi/2.
        _pyr0.pyr[band_id].real = ene*np.cos(ang0 + addcomp)
        _pyr0.pyr[band_id].imag = ene*np.sin(ang0 + addcomp)
        """
        
        
        if Luminance_mode:
            
            if b < pyr0.numBands()/2:
                _pyr0.pyr[band_id].real = 2.*_pyr0.pyr[band_id].imag
            else:
                _pyr0.pyr[band_id].real = -2.*_pyr0.pyr[band_id].imag
            _pyr0.pyr[band_id].imag *= 0.
    
        else:
            for i in xrange(3):
                if b < pyr0.numBands()/2:
                    pyrList_shift[i].pyr[band_id].real = 2.*pyrList_shift[i].pyr[band_id].imag
                else:
                    pyrList_shift[i].pyr[band_id].real = -2.*pyrList_shift[i].pyr[band_id].imag
                pyrList_shift[i].pyr[band_id].imag *= 0.
        

"""
#for debug
pyrcpy = copy.deepcopy(_pyr0)
out = pyrcpy.reconPyr()

cv2.imshow("phase_mod image",out/255.+0.5)
#cv2.imshow("phase_mod image",out/255.)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#%%
lamda = 2**(np.arange(pyr0.spyrHt()+2))
    
cosine = np.abs(np.cos(np.pi*np.arange(olevels).astype(np.float)/olevels))
    
print lamda
print cosine

#%%
maxval = disp.max()
minval = disp.min()
disp = (disp-minval)/float(maxval-minval)
        
#cv2.imshow("disp",disp)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

disp = disp_range*(disp-0.5)
#disp_pyr = SGFpyr(disp,sflevels-2)
disp_pyr2 = SGpyr(disp,sflevels-2)
G_d = []
G_d.append(0)#high-pass residual
for sf in range(sflevels-2):
    """
    div = 4.**(sf)
    
    tmp = disp_pyr.pyr[sf+1]/div
    minval = tmp.min()
    maxval = tmp.max()
    print maxval, minval
    if sf==0:
        sminval = -0.5#minval
        smaxval = 0.5#maxval
    
    
    tmp = (tmp-sminval)/float(smaxval-sminval)
    cv2.imshow("disp",tmp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    
    div = 2.**sf
    
    tmp = disp_pyr2.pyr[sf+1]/div
    
    #minval = tmp.min()
    #maxval = tmp.max()
    #print maxval, minval
    
    #tmp = (tmp-sminval)/float(smaxval-sminval)
    #cv2.imshow("disp",tmp)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    G_d.append(tmp)
            
    #fname = "test" + str(sf+1) + ".png"
    #cv2.imwrite(fname,np.uint8((_low+disp_range/2.)/disp_range*255.))

          
    
dmax_band = []
dmax_band.append(0)#high-pass residual
for sf in range(1,sflevels-1):
            
    _ori_band = []
    for ori in range(pyr0.numBands()):
        tmpD = G_d[sf].copy()
        
        #dmax = lamda[sf+1]/(4*cosine[ori])
        dmax = lamda[sf+1]/4.
        tmpD[tmpD>dmax]=dmax
        tmpD[tmpD<-dmax]=-dmax
        _ori_band.append(tmpD)
    
    dmax_band.append(_ori_band)
    
#%% 
print "weighting subband images..."
            
for s in range(pyr0.spyrHt()):
    #_CoG = cv2.resize(CoG,(CoG.shape[1]/(2**s),CoG.shape[0]/(2**s)),interpolation=cv2.INTER_LINEAR)
    for b in range(pyr0.numBands()):
        
        band_id = ((s*pyr0.numBands())+b)+1
        
        if Luminance_mode:
            A=np.tan(np.pi*dmax_band[s+1][b]*cosine[b]/lamda[s+1+1])
            _pyr0.pyr[band_id] *= A
        else:
            for i in xrange(3):
                A=np.tan(np.pi*dmax_band[s+1][b]*cosine[b]/lamda[s+1+1])
                pyrList_shift[i].pyr[band_id] *= A
        
    
if Luminance_mode:
    _pyr0.pyr[0] *= 0.
    _pyr0.pyr[len(pyr0.pyrSize)-1] *= 0.
else:
    for i in xrange(3):
        pyrList_shift[i].pyr[0] *= 0.
        pyrList_shift[i].pyr[len(pyr0.pyrSize)-1] *= 0.

#%%
print "reconstructing quadrature-phase pattern"

if Luminance_mode:
    out = _pyr0.reconPyr()
    
else:
    out = np.zeros(original.shape)
    for i in xrange(3):
        out[:,:,i] = pyrList_shift[i].reconPyr()
        
L=np.float64(original)+out
R=np.float64(original)-out


#clipping disparity inducer
clip_top_L = L - DR_high
clip_top_R = DR_low - R 
clip_top = clip_top_L
clip_top[clip_top_L<clip_top_R]=clip_top_R[clip_top_L<clip_top_R]
clip_top[clip_top<0.]=0.

clip_bottom_L = L - DR_low
clip_bottom_R = DR_high - R 
clip_bottom = clip_bottom_L
clip_bottom[clip_bottom_L>clip_bottom_R]=clip_bottom_R[clip_bottom_L>clip_bottom_R]
clip_bottom[clip_bottom>0.]=0.

out -= clip_bottom
out -= clip_top


#cv2.imwrite("clip_b.png",np.abs(clip_bottom))
#cv2.imwrite("clip_t.png",np.abs(clip_top))



L=np.float64(original)+out
R=np.float64(original)-out
    

maxval = L.max()#/255.
maxval2 = R.max()#/255.
if maxval2>maxval:
    maxval = maxval2
if maxval < 255.:
    maxval = 255.
    
minval = L.min()#/255.
minval2 = R.min()#/255.
if minval2<minval:
    minval = minval2
if minval > 0.:
    minval = 0.
print maxval, minval, minval2

#if compress_output:
    
out_L = (np.float64(L)-minval)/(maxval-minval)*255.
out_R = (np.float64(R)-minval)/(maxval-minval)*255.

#else:
#    if maxval > DR_high or minval > DR_low:
#        print "caution: clipped!!"
#    L[L>DR_high]=DR_high
#    L[L<DR_low]=DR_low
#    R[R>DR_high]=DR_high
#    R[R<DR_low]=DR_low   
#    #out_L = np.uint8((np.float32(L)-DR_low)/(DR_high-DR_low)*255.)
#    #out_R = np.uint8((np.float32(R)-DR_low)/(DR_high-DR_low)*255.)
#    out_L = (np.float64(L)-DR_low)/(DR_high-DR_low)*255.
#    out_R = (np.float64(R)-DR_low)/(DR_high-DR_low)*255.

if Luminance_mode:
    if HandleGamma:
        #out_L = np.uint8(((out_L/255.)**(1./gamma))*255.)
        #out_R = np.uint8(((out_R/255.)**(1./gamma))*255.)
        out_L=rgb2srgb(out_L/255.)*255.
        out_R=rgb2srgb(out_R/255.)*255.
    yuv0[:,:,0]=out_L
    rgb_L = cv2.cvtColor(yuv0, cv2.COLOR_YUV2BGR)
    yuv0[:,:,0]=out_R
    rgb_R = cv2.cvtColor(yuv0, cv2.COLOR_YUV2BGR)
else:
    rgb_L = out_L
    rgb_R = out_R
    
    if HandleGamma:
        #rgb_L = np.uint8(((rgb_L/255.)**(1./gamma))*255.)
        #rgb_R = np.uint8(((rgb_R/255.)**(1./gamma))*255.)
        rgb_L=rgb2srgb(rgb_L/255.)*255.
        rgb_R=rgb2srgb(rgb_R/255.)*255.
    


absadd=np.abs(out)
print absadd.max()
max_val=absadd.max()
cv2.imwrite("disp_inducer.png",np.uint8(out.real/max_val*127.5+127.5))
#cv2.imwrite("disp_inducer_nega.png",np.uint8(out.real/max_val*-127.5+127.5))

cv2.imwrite("hidden_L.png",rgb_L)
cv2.imwrite("hidden_R.png",rgb_R)

elapsed_time = time.time() - start
print ("elapsed time:{0}".format(elapsed_time)) + "(sec)"

