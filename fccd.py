#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

#import scipy.constants
 
# fccd readout
nbcol = 12 # number of colums/block
nbpcol=10 # real number of pixels/block
nb = 16 #number of blocks
nmux=12 # number of mux
nbmux  = nb * nmux # Nr. of ADC channels

# physical properties
#hnu = E* 1.15e-2 # single photon in ADU (approx)
ccd_pixel=30e-6  # 30 microns

# shifting/cropping frames
nrcols=484

ngcols=480
heigth=ngcols*2
width = heigth

nrows = 520
#gap = 33
gap = 34
nrows1=nrows-gap # good rows

# coordinates in the clean image
xx=np.linspace(-1,1,2*ngcols)

def clockXblocks1(data):
    """Translates `row` format to `clock` format."""
    return np.reshape(np.transpose(np.reshape(data,(nrows1, nbcol, nbmux), order='F'),[1,0,2]), (nrows1 * nbcol, nbmux), order='F')

def blocksXtif1(data):
    """Translates `ccd` format to `row` format."""
    #return np.concatenate((np.rot90(data[nrows1+gap*2:2*(nrows1)+gap*2,:],2),data[:nrows1,:]),axis=1)
    #return np.concatenate((np.rot90(data[nrows1+1+gap*2:2*(nrows1)+gap*2-1,:],2),data[2:nrows1,:]),axis=1)
    #return np.concatenate((np.rot90(data[nrows1+1+gap*2:2*(nrows1)+gap*2-1,:],2),data[1:nrows1-1,:]),axis=1)
    return np.concatenate((np.rot90(data[nrows1+gap*2:2*(nrows1)+gap*2-2,:],2),data[1:nrows1-1,:]),axis=1)

def bblocksXtif1(data): # stack the blocks
    return np.reshape(blocksXtif1(data),(nrcols,nbmux,nbcol))
    #return np.reshape(blocksXtif1(data),(nbmux,nrcols,nbcol))


def tif1Xbblocks(data): # tif from stacked blocks 
    return np.reshape(data[:,:,1:nbcol-1],(nrcols,nbmux*nbpcol))

def imgXtif1(data): # final image from tif1
    #return np.concatenate((data[6:486,0:960],np.rot90(data[6:486,960:],2)))
    #return np.concatenate((data[5:485,0:960],np.rot90(data[5:485,960:],2)))
    return np.concatenate((data[4:484,0:960],np.rot90(data[4:484,960:],2)))

        
# combine double exposure
def combine(data0, data1, t12, thres=3e3):
    msk=data0<thres
    return (t12+1)*(data0*msk+data1)/(t12*msk+1)

######################3
# denoise bblocks
bpts=60//2
gg=np.exp(-(np.arange(-bpts//2,bpts//2)/(bpts/4))**2)
gg/=np.sum(gg)
#gg=np.reshape(gg,(bpts,1))

def conv2d(data,filt):
    data_s=np.empty(np.shape(data))
    nr=data.shape[1]

    for r in range(nr):
        data_s[:,r] = np.convolve(data[:,r], gg, 'same')
    return data_s

def filter_bblocks(data):
    #yy=np.reshape(data[:,:,0],(nrcols,nbmux))
    # vertical stripes
    yy=np.reshape(data[:,:,11],(nrcols,192))
    # clip and smooth
    bkgthr=5 # background threshold
    yy_s=conv2d(np.clip(yy,-bkgthr,bkgthr),gg)
    yy_s=np.reshape(yy_s,(nrcols,nbmux,1))

    data_out = data-yy_s
    #data_out = data#-yy_s
    ###yy_avg=np.reshape(np.average(np.clip(bblocksXtif1(data_out)[1:11,:,:],0,2*bkgthr),axis=0),(1,192,12))
    yy_avg=np.reshape(np.average(np.clip(data_out[1:10,:,:],0,2*bkgthr),axis=0),(1,192,12))
    data_out -= yy_avg
    data_out *= data_out>7
    return data_out#-yy_s-yy_avg

######################3

def imgXraw_nofilter(data): # combine operations
    return imgXtif1(tif1Xbblocks(bblocksXtif1(data)))
  #  return imgXtif1(tif1Xbblocks(filter_bblocks(bblocksXtif1(data))))


def imgXraw(data): # combine operations
    #return imgXraw_nofilter(data)
    return imgXtif1(tif1Xbblocks(filter_bblocks(bblocksXtif1(data))))


