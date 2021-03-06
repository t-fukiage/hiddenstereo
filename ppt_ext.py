#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:11:48 2017

@author: taikifukiage
"""
import numpy

from pyPyrTools.pyramid import pyramid
from pyPyrTools.sp0Filters import sp0Filters
from pyPyrTools.maxPyrHt import maxPyrHt
from pyPyrTools.corrDn import corrDn

from pyPyrTools.SFpyr import SFpyr
from pyPyrTools.steer2HarmMtx import steer2HarmMtx
from pyPyrTools.rcosFn import rcosFn
from pyPyrTools.pointOp import pointOp
import scipy
from pyPyrTools.mkAngle import mkAngle
import cmath

class SCFpyr_mod(SFpyr):
    filt = ''
    edges = ''
    
    #constructor
    def __init__(self, *args):    # (image, height, order, twidth)
        self.pyrType = 'steerableFrequency'

        if len(args) > 0:
            self.image = args[0]
        else:
            print "First argument (image) is required."
            return

        #------------------------------------------------
        # defaults:

        max_ht = numpy.floor( numpy.log2( min(self.image.shape) ) ) - 2
        if len(args) > 1:
            if(args[1] > max_ht):
                print "Error: cannot build pyramid higher than %d levels." % (max_ht)
            ht = args[1]
        else:
            ht = max_ht
        ht = int(ht)
            
        if len(args) > 2:
            if args[2] > 15 or args[2] < 0:
                print "Warning: order must be an integer in the range [0,15]. Truncating."
                order = min( max(args[2],0), 15 )
            else:
                order = args[2]
        else:
            order = 3

        nbands = order+1

        if len(args) > 3:
            if args[3] <= 0:
                print "Warning: twidth must be positive. Setting to 1."
                twidth = 1
            else:
                twidth = args[3]
        else:
            twidth = 1

        #------------------------------------------------------
        # steering stuff:

        if nbands % 2 == 0:
            harmonics = numpy.array(range(nbands/2)) * 2 + 1
        else:
            harmonics = numpy.array(range((nbands-1)/2)) * 2

        steermtx = steer2HarmMtx(harmonics,
                                 numpy.pi*numpy.array(range(nbands))/nbands,
                                 'even')
        #------------------------------------------------------
        
        dims = numpy.array(self.image.shape)
        ctr = numpy.ceil((numpy.array(dims)+0.5)/2)
        
        (xramp, yramp) = numpy.meshgrid((numpy.array(range(1,dims[1]+1))-ctr[1])/
                                     (dims[1]/2), 
                                     (numpy.array(range(1,dims[0]+1))-ctr[0])/
                                     (dims[0]/2))
        angle = numpy.arctan2(yramp, xramp)
        log_rad = numpy.sqrt(xramp**2 + yramp**2)
        log_rad[int(ctr[0])-1, int(ctr[1])-1] = log_rad[int(ctr[0])-1, int(ctr[1])-2]
        log_rad = numpy.log2(log_rad);

        ## Radial transition function (a raised cosine in log-frequency):
        (Xrcos, Yrcos) = rcosFn(twidth, (-twidth/2.0), numpy.array([0,1]))
        Yrcos = numpy.sqrt(Yrcos)

        YIrcos = numpy.sqrt(1.0 - Yrcos**2)
        lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)

        imdft = numpy.fft.fftshift(numpy.fft.fft2(self.image))

        self.pyr = []
        self.pyrSize = []

        hi0mask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)

        hi0dft = imdft * hi0mask.reshape(imdft.shape[0], imdft.shape[1])
        hi0 = numpy.fft.ifft2(numpy.fft.ifftshift(hi0dft))

        self.pyr.append(numpy.real(hi0.copy()))
        self.pyrSize.append(hi0.shape)

        lo0mask = lo0mask.reshape(imdft.shape[0], imdft.shape[1])
        lodft = imdft * lo0mask

        for i in range(ht):
            bands = numpy.zeros((lodft.shape[0]*lodft.shape[1], nbands))
            bind = numpy.zeros((nbands, 2))
        
            Xrcos -= numpy.log2(2)

            lutsize = 1024
            Xcosn = numpy.pi * numpy.array(range(-(2*lutsize+1), (lutsize+2))) / lutsize

            order = nbands -1
            const = (2**(2*order))*(scipy.misc.factorial(order, exact=True)**2)/float(nbands*scipy.misc.factorial(2*order, exact=True))

            alfa = ( (numpy.pi+Xcosn) % (2.0*numpy.pi) ) - numpy.pi
            Ycosn = ( 2.0*numpy.sqrt(const) * (numpy.cos(Xcosn)**order) * 
                      (numpy.abs(alfa)<numpy.pi/2.0).astype(int) )
            log_rad_tmp = numpy.reshape(log_rad, (1,log_rad.shape[0]*
                                                  log_rad.shape[1]))
            himask = pointOp(log_rad_tmp, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)
            
            himask = himask.reshape(lodft.shape[0], lodft.shape[1])
            for b in range(nbands):
                angle_tmp = numpy.reshape(angle, 
                                          (1,angle.shape[0]*angle.shape[1]))
                anglemask = pointOp(angle_tmp, Ycosn,
                                    Xcosn[0]+numpy.pi*b/nbands, 
                                    Xcosn[1]-Xcosn[0], 0)
                anglemask = anglemask.reshape(lodft.shape[0], lodft.shape[1])
                banddft = (cmath.sqrt(-1)**order) * lodft * anglemask * himask
                band = numpy.negative(numpy.fft.ifft2(numpy.fft.ifftshift(banddft)))
                self.pyr.append(band.copy())
                self.pyrSize.append(band.shape)

            dims = numpy.array(lodft.shape)
            ctr = numpy.ceil((dims+0.5)/2)
            lodims = numpy.ceil((dims-0.5)/2)
            loctr = numpy.ceil((lodims+0.5)/2)
            lostart = ctr - loctr
            loend = lostart + lodims

            log_rad = log_rad[int(lostart[0]):int(loend[0]), int(lostart[1]):int(loend[1])]
            angle = angle[int(lostart[0]):int(loend[0]), int(lostart[1]):int(loend[1])]
            lodft = lodft[int(lostart[0]):int(loend[0]), int(lostart[1]):int(loend[1])]
            YIrcos = numpy.abs(numpy.sqrt(1.0 - Yrcos**2))
            log_rad_tmp = numpy.reshape(log_rad, 
                                        (1,log_rad.shape[0]*log_rad.shape[1]))
            lomask = pointOp(log_rad_tmp, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0],
                             0)
            lodft = lodft * lomask.reshape(lodft.shape[0], lodft.shape[1])

        lodft = numpy.fft.ifft2(numpy.fft.ifftshift(lodft))
        self.pyr.append(numpy.real(numpy.array(lodft).copy()))
        self.pyrSize.append(lodft.shape)

    # methods
    def reconPyr(self, *args):
        if len(args) > 0:
            levs = args[0]
        else:
            levs = 'all'

        if len(args) > 1:
            bands = args[1]
        else:
            bands = 'all'

        if len(args) > 2:
            if args[2] <= 0:
                print "Warning: twidth must be positive. Setting to 1."
                twidth = 1
            else:
                twidth = args[2]
        else:
            twidth = 1

        #-----------------------------------------------------------------

        pind = self.pyrSize
        #Nsc = int(numpy.log2(pind[0][0] / pind[-1][0]))
        Nsc = int(numpy.round(numpy.log2(pind[0][0] / pind[-1][0])))
        Nor = (len(pind)-2) / Nsc
        

        pyrIdx = 1
        for nsc in range(Nsc):
            firstBnum = nsc * Nor+2
            dims = pind[firstBnum][:]
            ctr = (numpy.ceil((dims[0]+0.5)/2.0), numpy.ceil((dims[1]+0.5)/2.0)) #-1?
            ang = mkAngle(dims, 0, ctr)
            ang[int(ctr[0])-1, int(ctr[1])-1] = -numpy.pi/2.0
            for nor in range(Nor):
                nband = nsc * Nor + nor + 1
                ch = self.pyr[nband]
                ang0 = numpy.pi * nor / Nor
                xang = ((ang-ang0+numpy.pi) % (2.0*numpy.pi)) - numpy.pi
                amask = 2 * (numpy.abs(xang) < (numpy.pi/2.0)).astype(int) + (numpy.abs(xang) == (numpy.pi/2.0)).astype(int)
                amask[int(ctr[0])-1, int(ctr[1])-1] = 1
                amask[:,0] = 1
                amask[0,:] = 1
                amask = numpy.fft.fftshift(amask)
                ch = numpy.fft.ifft2(amask * numpy.fft.fft2(ch))  # 'Analytic' version
                # f = 1.000008  # With this factor the reconstruction SNR
                                # goes up around 6 dB!
                f = 1
                ch = f*0.5*numpy.real(ch)   # real part
                self.pyr[pyrIdx] = ch
                pyrIdx += 1

        res = self.reconSFpyr(levs, bands, twidth);

        return res


class SGpyr(pyramid):
    
    
    filt = ''
    edges = ''
    
    #constructor
    def __init__(self, *args):    # (image height, filter file, edges)
        #self.pyrType = 'steerable'
        if len(args) > 0:
            self.image = numpy.array(args[0])
        else:
            print "First argument (image) is required."
            return

        #------------------------------------------------
        # defaults:

        filters = sp0Filters()

        harmonics = filters['harmonics']
        lo0filt = filters['lo0filt']
        hi0filt = filters['hi0filt']
        lofilt = filters['lofilt']
        bfilts = filters['bfilts']
        steermtx = filters['mtx']
        
        print lo0filt.sum()
        print lofilt.sum()
        
        max_ht = maxPyrHt(self.image.shape, lofilt.shape)
        if len(args) > 1:
            if args[1] == 'auto':
                ht = max_ht
            elif args[1] > max_ht:
                print "Error: cannot build pyramid higher than %d levels." % (
                    max_ht)
                return
            else:
                ht = args[1]
        else:
            ht = max_ht

        if len(args) > 2:
            edges = args[2]
        else:
            edges = 'reflect1'

        #------------------------------------------------------

        nbands = bfilts.shape[1]

        self.pyr = []
        self.pyrSize = []
        for n in range((ht*nbands)+2):
            self.pyr.append([])
            self.pyrSize.append([])

        im = self.image
        im_sz = im.shape
        pyrCtr = 0

        hi0 = corrDn(image = im, filt = hi0filt, edges = edges);

        self.pyr[pyrCtr] = hi0
        self.pyrSize[pyrCtr] = hi0.shape

        pyrCtr += 1

        lo = corrDn(image = im, filt = lo0filt, edges = edges)
        for i in range(ht):
            self.pyr[pyrCtr] = numpy.array(lo)
            self.pyrSize[pyrCtr] = (lo.shape[0], lo.shape[1])
            pyrCtr += 1

            lo = corrDn(image = lo, filt = lofilt, edges = edges, step = (2,2))

        self.pyr[pyrCtr] = numpy.array(lo)
        self.pyrSize[pyrCtr] = lo.shape
        
