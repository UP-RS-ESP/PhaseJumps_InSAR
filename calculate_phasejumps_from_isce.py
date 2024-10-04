import os, logging, sys, glob, tqdm, argparse
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

import isce
from imageMath import IML

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# Standard version (no numba)
def abs_gradient(data):
    # get phase from complex data
    int_phase = np.angle(data)
    # calculate diff and store in array
    abs_grad = np.abs(np.diff(int_phase, axis=0, prepend=0)).astype(np.float32)
    return np.array([np.median(abs_grad, axis=1).astype(np.float32), np.std(abs_grad, axis=1).astype(np.float32)]).T


def abs_gradient_forloop(data):
    # get phase from complex data
    int_phase = np.angle(data).astype(np.float32)
    # calculate diff and store in array
    array = np.concatenate((np.zeros((1, int_phase.shape[1]), dtype=np.float32), 
                            int_phase.astype(np.float32))).astype(np.float32)
    
    grad_az = np.diff(array.T).T.astype(np.float32)
    grad_az_stats = np.empty( (grad_az.shape[0], 2), dtype=np.float32)
    for n in range(grad_az.shape[0]):
        grad_az_stats[n,0] = np.median(grad_az[n,:])
        grad_az_stats[n,1] = np.std(grad_az[n,:])
    return grad_az_stats

# using GPU via cupy
def abs_gradient_complex_gpu(data):
    data_gpu = cp.asarray(data)
    # get phase from complex data
    int_phase = cp.angle(data_gpu)
    # calculate diff and store in array
    abs_grad = cp.abs(cp.diff(int_phase, axis=0, prepend=0))
    data_gpu = None
    return cp.nanmedian(abs_grad, axis=1).get().astype(cp.float32), cp.nanstd(abs_grad, axis=1).get().astype(np.float32)


def abs_gradient_float_gpu(data):
    data_gpu = cp.asarray(data)
    # calculate diff and store in array
    abs_grad = cp.abs(cp.diff(data_gpu, axis=0, prepend=0))
    data_gpu = None
    return cp.nanmedian(abs_grad, axis=1).get().astype(cp.float32), cp.nanstd(abs_grad, axis=1).get().astype(np.float32)



# # not using numba version - only useful if data are loaded as stack
# import numba as nb
# @nb.njit(parallel=True)
# def abs_gradient_numba(data):
#     # get phase from complex data
#     int_phase = np.angle(data).astype(np.float32)
#     # calculate diff and store in array
#     array = np.concatenate((np.zeros((1, int_phase.shape[1]), dtype=np.float32), 
#                             int_phase.astype(np.float32))).astype(np.float32)
#     
#     grad_az = np.diff(array.T).T.astype(np.float32)
#     grad_az_stats = np.empty( (grad_az.shape[0], 2), dtype=np.float32)
#     for n in nb.prange(grad_az.shape[0]):
#         grad_az_stats[n,0] = np.median(grad_az[n,:])
#         grad_az_stats[n,1] = np.std(grad_az[n,:])
#     return grad_az_stats
# 
# 
# @nb.njit(parallel=True)
# def diff_along_azimuth(ds_unw):
#     #ds_grad_az in shape (time,rows-Y,cols-X)
#     ds_grad_az = np.empty(ds_unw.shape, dtype=np.float32)
#     for n in nb.prange(ds_unw.shape[0]):
#         array = np.concatenate((np.zeros((1, ds_unw.shape[2]), dtype=np.float32), 
#                                 ds_unw[n,:,:].astype(np.float32))).astype(np.float32)
#         
#         ds_grad_az[n,:,:] = np.diff(array.T).T.astype(np.float32)
#     return ds_grad_az


path='/raid-gpu2/InSAR/Olkaria/S1_tr130_asc/NaivashaNakuru_COP_az2_rg7/merged/interferograms/'
ifg_file='filt_fine.int.xml'
# uncomment/comment depending on what file type you are using
#ifg_file='filt_fine.unw.xml'
logging.info('Getting filelist from %s'%path)
ifg_date_fn = glob.glob(os.path.join(path, '*', ifg_file))
ifg_date_fn.sort()

nr_ifg_files = len(ifg_date_fn)
logging.info('Number of files: %d'%nr_ifg_files)

# adjust png output filename
ph_jump_fn = 'NaivashaNakuru_COP_az2_rg7_unwrapped_phasejumps.png'

#get size of template array
inname = os.path.join(ifg_date_fn[0])
img, dataname, metaname = IML.loadImage(inname)
img_width = img.getWidth()
img_length = img.getLength()
img = None

# create large array to store gradient results
logging.info('Creating array with %d x %d x %d dimensions for storing gradient statistics median and std. dev. for each row'%(nr_ifg_files, img_length, 2) )
abs_grad_stats = np.empty( (nr_ifg_files, img_length, 2), dtype=np.float32)

logging.info('Loading phase files and calculating gradients. Storing only median gradient and standard deviation for each row.')
for i in tqdm.tqdm(range(len(ifg_date_fn))):
    inname = os.path.join(ifg_date_fn[i])
    img, dataname, metaname = IML.loadImage(inname)
    if img.dataType == 'FLOAT':
        #unwrapped data
        data = img.memMap()[:,1,:].astype(np.float32)
        abs_grad_stats[i,:,0], abs_grad_stats[i,:,1] = abs_gradient_float_gpu(data)

    elif img.dataType == 'CFLOAT':
        data = np.squeeze(img.memMap()).astype(np.complex64)
        abs_grad_stats[i,:,0], abs_grad_stats[i,:,1] = abs_gradient_complex_gpu(data)

    #abs_grad_stats[i,:] = abs_gradient(data)
    #numpy for loop appears to be fastest option
    #abs_grad_stats[i,:,0], abs_grad_stats[i,:,1] = abs_gradient_gpu(data)
    #abs_grad_stats[i,:,:] = abs_gradient_forloop(data)
    #abs_grad_stats[i,:,:] = abs_gradient_numba(data)

fg, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(14,8), dpi=300)
im0 = ax[0].imshow(abs_grad_stats[:,:,0], cmap=plt.cm.viridis, 
                   vmin=np.percentile(abs_grad_stats[:,:,0],2),
                   vmax=np.percentile(abs_grad_stats[:,:,0],98))
ax[0].set_title('Median Phase jump magnitude from interferogram')
ax[0].set_xlabel('Ifg number (date)')
ax[0].set_ylabel('Y or Azimuth (radar coordinates)')
h = plt.colorbar(im0, ax=ax[0], orientation='horizontal')

im1 = ax[1].imshow(abs_grad_stats[:,:,1], cmap=plt.cm.magma,
                   vmin=np.percentile(abs_grad_stats[:,:,1],2),
                   vmax=np.percentile(abs_grad_stats[:,:,1],98))
ax[1].set_title('Std. Dev. of Phase jump magnitude from interferogram')
ax[1].set_xlabel('Ifg number (date)')
ax[1].set_ylabel('Y or Azimuth (radar coordinates)')
h = plt.colorbar(im1, ax=ax[1], orientation='horizontal')

fg.tight_layout()
fg.savefig(ph_jump_fn, dpi=300)

