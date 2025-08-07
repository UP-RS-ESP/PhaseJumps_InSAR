#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate correlation statistics to identify days with low coherence or with no phase values
Authors: Bodo Bookhagen
bodo.bookhagen@uni-potsdam.de
V0.1 July-2025
"""

import os, logging, sys, glob, tqdm, argparse, datetime
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import isce
from imageMath import IML

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

synopsis = "Extract azimuth gradient from unwrapped phase in radar coordinates and find phase jumps"
EXAMPLE = """example:
    python calculate_phasejumps_gpu_from_isce.py \
            --ifg_path /raid-gpu2/InSAR/Olkaria/S1_tr130_asc/Olkaria2_COP_az2_rg7/merged_rg20az4/interferograms/ \
            --burst-number 2 \
            --global_stats_csv Olkaria_rg20_az4_global_stats.csv \
            --Phase_gradient_PNG Olkaria_rg20_az4_phasegradient_ts.png
    
    References
    1) Wang et al., 2017 "Improving burst aligment in tops interferometry with BESD",
       10.1109/LGRS.2017.2767575
    2) Zhong et al., 2014 "A Quality-Guided and Local Minimum Discontinuity Based 
       Phase Unwrapping Algorithm for InSAR InSAS Interograms", 10.1109/LGRS.2013.2252880
"""

DESCRIPTION = """
Phase gradient calculation and burst determination

Oct-2024, Sofia Viotto, Bodo Bookhagen (bodo.bookhagen@uni-potsdam.de)
"""


def cmdLineParser():
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(
        description=DESCRIPTION, epilog=EXAMPLE, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--ifg_path",
        help='Path to the root directory contains the unwrapped interferograms and coherence files. Usually this is "<path>/merged/interferograms/"',
        required=True,
    )
    parser.add_argument(
        "--ifg_file",
        default="filt_fine.unw.xml",
        help='Filename of filtered wrapped or unwrapped interferogram. Default is "filt_fine.unw.xml". It is assumed that coherence file is called "filt_fine.cor.xml".',
        required=True,
    )
    parser.add_argument(
        "--burst_number",
        help='Number of burst overlaps. Can be determined from ESD or other directories. For example, "ls -1 ESD/20240717_20240729/IW3/overlap_??.int | wc -l"',
        required=True,
    )
    parser.add_argument(
        "--global_stats_csv",
        default="global_phasegrad_coherence_stats.csv",
        help="CSV filename to save global statistics containing dates, phase gradients, and coherence statistics.",
        required=False,
    )
    parser.add_argument(
        "--Phase_gradient_PNG",
        help="Plot showing the median phase gradient as time series",
        required=False,
    )
    parser.add_argument(
        "--wavelength",
        default=0.05546576,
        help="Radar wavelength for conversion to mm",
        required=False,
    )
    return parser.parse_args()


# Standard version (no numba)
def abs_gradient(data):
    # get phase from complex data
    int_phase = np.angle(data)
    # calculate diff and store in array
    abs_grad = np.abs(np.diff(int_phase, axis=0, prepend=0)).astype(np.float32)
    return np.array(
        [
            np.median(abs_grad, axis=1).astype(np.float32),
            np.std(abs_grad, axis=1).astype(np.float32),
        ]
    ).T


def abs_gradient_forloop(data):
    # get phase from complex data
    int_phase = np.angle(data).astype(np.float32)
    # calculate diff and store in array
    array = np.concatenate(
        (
            np.zeros((1, int_phase.shape[1]), dtype=np.float32),
            int_phase.astype(np.float32),
        )
    ).astype(np.float32)

    grad_az = np.diff(array.T).T.astype(np.float32)
    grad_az_stats = np.empty((grad_az.shape[0], 2), dtype=np.float32)
    for n in range(grad_az.shape[0]):
        grad_az_stats[n, 0] = np.median(grad_az[n, :])
        grad_az_stats[n, 1] = np.std(grad_az[n, :])
    return grad_az_stats


# using GPU via cupy
def abs_gradient_severity_complex_gpu(ifg, cor, phase2range):
    ifg_gpu = cp.asarray(ifg)
    cor_gpu = cp.asarray(cor)
    phase2range_gpu = cp.asarray(phase2range)

    # get phase from complex data
    int_phase = cp.angle(ifg_gpu)
    # calculate diff and store in array
    del ifg_gpu
    abs_grad *= phase2range
    abs_grad *= 1000
    abs_grad[cor_gpu < 0.75] = cp.nan
    cor_median = np.round(np.median(cor_gpu).astype(cp.float32), 3)
    cor_mean = np.round(cp.nanmean(cor_gpu).astype(cp.float32), 3)
    cor_std = np.round(cp.nanstd(cor_gpu).astype(cp.float32), 3)
    del cor_gpu
    median_gpu = cp.nanmedian(abs_grad)
    median = np.round(median_gpu.astype(cp.float32), 2)
    mean = np.round(cp.nanmean(abs_grad).astype(cp.float32), 2)
    std = np.round(cp.nanstd(abs_grad).astype(cp.float32), 2)
    medianv = np.round(cp.nanmedian(abs_grad, axis=1).get().astype(cp.float32), 2)
    meanv = np.round(cp.nanmean(abs_grad, axis=1).get().astype(cp.float32), 2)
    stdv = np.round(cp.nanmedian(abs_grad, axis=1).get().astype(cp.float32), 2)

    # calculate severity
    severity = np.round(cp.divide(abs_grad, median_gpu).get().astype(cp.float32), 0)
    severity[severity > 1] = 1
    severity_alongX = np.nansum(severity, axis=1)
    nonan_counts = np.count_nonzero(~np.isnan(severity), axis=1)
    nonan_counts = nonan_counts.astype(int)
    # Remove coordinates that are smaller than a certain percentage of NoNanPixels
    nonan_counts_p10 = np.nanpercentile(nonan_counts, 10)
    severity_alongX[nonan_counts < nonan_counts_p10] = np.nan
    severity_alongX_proportion = np.divide(severity_alongX, nonan_counts) * 100
    # Set first two rows and the two last rows to nan to avoid phase_jumps at the border of the dataset
    severity_alongX_proportion = np.round(severity_alongX_proportion, 1)
    del severity, severity_alongX, nonan_counts, nonan_counts_p10
    del abs_grad, phase2range_gpu
    return (
        medianv,
        meanv,
        stdv,
        median,
        mean,
        std,
        cor_median,
        cor_mean,
        cor_std,
        severity_alongX_proportion,
    )


def abs_gradient_severity_float_gpu(ifg, cor, phase2range):
    ifg_gpu = cp.asarray(ifg)
    cor_gpu = cp.asarray(cor)
    phase2range_gpu = cp.asarray(phase2range)

    # calculate diff and store in array
    abs_grad = cp.abs(cp.diff(ifg_gpu, axis=0, prepend=0))
    del ifg_gpu
    abs_grad *= phase2range
    abs_grad *= 1000
    abs_grad[cor_gpu < 0.75] = cp.nan
    cor_median = np.round(np.median(cor_gpu).astype(cp.float32), 3)
    cor_mean = np.round(cp.nanmean(cor_gpu).astype(cp.float32), 3)
    cor_std = np.round(cp.nanstd(cor_gpu).astype(cp.float32), 3)
    del cor_gpu
    median_gpu = cp.nanmedian(abs_grad)
    median = np.round(median_gpu.astype(cp.float32), 2)
    mean = np.round(cp.nanmean(abs_grad).astype(cp.float32), 2)
    std = np.round(cp.nanstd(abs_grad).astype(cp.float32), 2)
    medianv = np.round(cp.nanmedian(abs_grad, axis=1).get().astype(cp.float32), 2)
    meanv = np.round(cp.nanmean(abs_grad, axis=1).get().astype(cp.float32), 2)
    stdv = np.round(cp.nanmedian(abs_grad, axis=1).get().astype(cp.float32), 2)

    # calculate severity
    severity = np.round(cp.divide(abs_grad, median_gpu).get().astype(cp.float32), 0)
    severity[severity > 1] = 1
    severity_alongX = np.nansum(severity, axis=1)
    nonan_counts = np.count_nonzero(~np.isnan(severity), axis=1)
    nonan_counts = nonan_counts.astype(int)
    # Remove coordinates that are smaller than a certain percentage of NoNanPixels
    nonan_counts_p10 = np.nanpercentile(nonan_counts, 10)
    severity_alongX[nonan_counts < nonan_counts_p10] = np.nan
    severity_alongX_proportion = np.divide(severity_alongX, nonan_counts) * 100
    # Set first two rows and the two last rows to nan to avoid phase_jumps at the border of the dataset
    severity_alongX_proportion = np.round(severity_alongX_proportion, 1)
    del severity, severity_alongX, nonan_counts, nonan_counts_p10
    del abs_grad, phase2range_gpu
    return (
        medianv,
        meanv,
        stdv,
        median,
        mean,
        std,
        cor_median,
        cor_mean,
        cor_std,
        severity_alongX_proportion,
    )


def save_global_stats2txt(global_stats_fn, refdates, secdates, global_stats_array):
    refList = [datetime.datetime.strptime(date, "%Y%m%d") for date in refdates]
    secList = [datetime.datetime.strptime(date, "%Y%m%d") for date in secdates]
    Bt = [(sec - ref).days for ref, sec in zip(refList, secList)]
    global_stats = np.c_[refdates, secdates, Bt, global_stats_array]
    header = "RefDate, SecDate, Dt_days, Phasegrad_median_mm, Phasegrad_mean_mm, Phasegrad_std_mm, Coherence_median, Coherence_mean, Coherence_std"
    np.savetxt(
        global_stats_fn,
        global_stats,
        fmt="%s",
        delimiter=",",
        newline="\n",
        header=header,
        footer="",
        comments="# ",
    )


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
#     grad_az_stats = np.empty( (grad_az.shape[0], 3), dtype=np.float32)
#     for n in nb.prange(grad_az.shape[0]):
#         grad_az_stats[n,0] = np.median(grad_az[n,:])
#         grad_az_stats[n,1] = np.mean(grad_az[n,:])
#         grad_az_stats[n,2] = np.std(grad_az[n,:])
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
#
#
# @nb.njit(parallel=True)
# def calc_severity(ds_abs_grad, coefficient):
#     severity = np.empty((ds_abs_grad.shape), dtype=np.int16)
#     for i in prange(ds_abs_grad.shape[0]):
#         out = np.round(np.divide(ds_abs_grad[i,:,:], coefficient[i]), 0)
#         severity[i,:,:] = np.int16(out)
#     return severity
#
#
# @nb.njit(parallel=True)
# def stats_ds_unw(ds_unw):
#     stats = np.empty((ds_unw.shape[0], 3), dtype=np.float32)
#     for n in nb.prange(ds_unw.shape[0]):
#         array = ds_unw[n, :, :]
#         stats[n, 0] = np.round(np.nanmedian(array), 2)
#         stats[n, 1] = np.round(np.nanmean(array), 2)
#         stats[n, 2] = np.round(np.nanstd(array), 2)
#         # stats[n,3]=np.count_nonzero(~np.isnan(array))
#     return stats


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    warnings.simplefilter("ignore")
    warnings.filterwarnings(action="ignore", message="Mean of empty slice")

    args = cmdLineParser()
    refdate_unique = np.unique(refdates)
    date_unique = np.unique(secdates)

    phase2range = float(args.WAVELENGTH) / (4.0 * np.pi)
    np.seterr(all="warn")
    np.seterr(invalid="ignore")

    path = args.path
    burst_number = args.burst_number

    logging.info("Getting filelist from %s" % path)
    ifg_date_fn = glob.glob(os.path.join(path, "*", ifg_file))
    ifg_date_fn.sort()

    nr_ifg_files = len(ifg_date_fn)
    logging.info("Number of files: %d" % nr_ifg_files)

    # get size of template array for first image
    inname = os.path.join(ifg_date_fn[0])
    img, dataname, metaname = IML.loadImage(inname)
    img_width = img.getWidth()
    img_length = img.getLength()
    img = None

    # create large array to store gradient results
    logging.info(
        "Creating array with %d x %d x %d dimensions for storing gradient statistics median and std. dev. for each row"
        % (nr_ifg_files, img_length, 3)
    )
    # no need to store all data in large array - this can be calculated for each date separately
    # ds_array = np.empty( (nr_ifg_files, img_length, img_width), dtype=np.float32)
    correlation_stats = np.empty((nr_ifg_files, img_length, 5), dtype=np.float32)
    correlation_stats.fill(np.nan)
    # phase_stats = np.empty( (nr_ifg_files, img_length, 5), dtype=np.float32)
    # create lists for storing dates
    refdates = []
    secdates = []

    logging.info(
        "Loading phase files and calculating gradients. Storing only median, mean, and standard deviation of phase gradient for each row and for each date. Calculating global statistics from coherence and phase gradient for each time step."
    )
    for i in tqdm.tqdm(range(len(ifg_date_fn)), desc="Loading data and calculating"):
        inname = os.path.join(ifg_date_fn[i])
        ifgimg, _, _ = IML.loadImage(inname)
        corimg, _, _ = IML.loadImage(inname[:-7] + "cor")
        date_dir = os.path.dirname(inname).split("/")[-1]
        refdate = date_dir.split("_")[0]
        refdates.append(refdate)
        secdate = date_dir.split("_")[1]
        secdates.append(secdate)
        if ifgimg.dataType == "FLOAT":
            # unwrapped data
            #            ifg = ifgimg.memMap()[:,1,:].astype(np.float32)
            cor = corimg.memMap()[:, :, 0].astype(np.float32)
            cor[cor == 0] = np.nan
            # get correlation statistics along X direction:
            cor_mean = np.nanmean(cor, axis=1)
            cor_median = np.nanmedian(cor, axis=1)
            cor_var = np.nanvar(cor, axis=1)
            cor_p25 = np.nanpercentile(cor, 0.25, axis=1)
            cor_p75 = np.nanpercentile(cor, 0.75, axis=1)
            correlation_stats[i, :] = np.c_[
                cor_mean, cor_median, cor_var, cor_p25, cor_p75
            ]
            # ifg[np.isnan(cor)] = np.nan
            # ifg *= phase2range
            # ds_array[i,:,:] = data
            # abs_grad_stats[i,:,0], abs_grad_stats[i,:,1], abs_grad_stats[i,:,2], abs_grad_global_stats[i,0], abs_grad_global_stats[i,1], abs_grad_global_stats[i,2], cor_global_stats[i,0], cor_global_stats[i,1], cor_global_stats[i,2], severity_alongX_proportion[i,:] = abs_gradient_severity_float_gpu(ifg, cor, phase2range)
        # elif img.dataType == 'CFLOAT':
        #     # ifg = np.squeeze(img.memMap()).astype(np.complex64)
        #     cor = corimg.memMap()[:,:,0].astype(np.float32)
        #     # abs_grad_stats[i,:,0], abs_grad_stats[i,:,1], abs_grad_stats[i,:,2], abs_grad_global_stats[i,0], abs_grad_global_stats[i,1], abs_grad_global_stats[i,2], cor_global_stats[i,0], cor_global_stats[i,1], cor_global_stats[i,2], severity_alongX_proportion[i,:] = abs_gradient_severity_complex_gpu(ifg, cor, phase2range)

        # abs_grad_stats[i,:] = abs_gradient(data)
        # numpy for loop appears to be fastest option
        # abs_grad_stats[i,:,0], abs_grad_stats[i,:,1] = abs_gradient_gpu(data)
        # abs_grad_stats[i,:,:] = abs_gradient_forloop(data)
        # abs_grad_stats[i,:,:] = abs_gradient_numba(data)

    refdate_unique = np.unique(np.int32(refdates))
    date_unique = np.unique(np.int32(secdates))
    refdate_unique_dt = []
    correlation_refdate_stats = np.empty((len(refdate_unique), 4), dtype=np.float32)
    correlation_refdate_stats.fill(np.nan)
    correlation_refdate_stats_top = np.empty((len(refdate_unique), 4), dtype=np.float32)
    correlation_refdate_stats_top.fill(np.nan)
    correlation_refdate_stats_bottom = np.empty(
        (len(refdate_unique), 4), dtype=np.float32
    )
    correlation_refdate_stats_bottom.fill(np.nan)
    for i in range(len(refdate_unique)):
        crefdate = refdate_unique[i]
        refdate_unique_dt.append(pd.to_datetime(str(crefdate)))
        (refidx,) = np.where(crefdate == np.int32(refdates))
        correlation_stats_max = np.nanmax(correlation_stats[refidx, :, 4])
        correlation_stats_min = np.nanmin(correlation_stats[refidx, :, 3])
        correlation_stats_mean = np.nanmean(correlation_stats[refidx, :, 1])
        correlation_stats_median = np.nanmedian(correlation_stats[refidx, :, 1])
        correlation_refdate_stats[i, :] = np.r_[
            correlation_stats_mean,
            correlation_stats_median,
            correlation_stats_min,
            correlation_stats_max,
        ]
        correlation_stats_max_top = np.nanmax(
            correlation_stats[refidx, 0 : int(correlation_stats.shape[1] / 2), 4]
        )
        correlation_stats_min_top = np.nanmin(
            correlation_stats[refidx, 0 : int(correlation_stats.shape[1] / 2), 3]
        )
        correlation_stats_mean_top = np.nanmean(
            correlation_stats[refidx, 0 : int(correlation_stats.shape[1] / 2), 1]
        )
        correlation_stats_median_top = np.nanmedian(
            correlation_stats[refidx, 0 : int(correlation_stats.shape[1] / 2), 1]
        )
        correlation_refdate_stats_top[i, :] = np.r_[
            correlation_stats_mean_top,
            correlation_stats_median_top,
            correlation_stats_min_top,
            correlation_stats_max_top,
        ]
        correlation_stats_max_bottom = np.nanmax(
            correlation_stats[refidx, int(correlation_stats.shape[1] / 2) :, 4]
        )
        correlation_stats_min_bottom = np.nanmin(
            correlation_stats[refidx, int(correlation_stats.shape[1] / 2) :, 3]
        )
        correlation_stats_mean_bottom = np.nanmean(
            correlation_stats[refidx, int(correlation_stats.shape[1] / 2) :, 1]
        )
        correlation_stats_median_bottom = np.nanmedian(
            correlation_stats[refidx, int(correlation_stats.shape[1] / 2) :, 1]
        )
        correlation_refdate_stats_bottom[i, :] = np.r_[
            correlation_stats_mean_bottom,
            correlation_stats_median_bottom,
            correlation_stats_min_bottom,
            correlation_stats_max_bottom,
        ]

    correlation_ts_fn = "S1_tr130_asc_OlkariaNakuru_COP_az8_rg40_pb100_ts.png"
    fg, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(14, 8), dpi=300, layout="constrained"
    )
    im0 = ax[0].imshow(
        correlation_stats[:, :, 1],
        cmap=plt.cm.viridis,
        vmin=np.nanpercentile(correlation_stats[:, :, 1], 2),
        vmax=np.nanpercentile(correlation_stats[:, :, 1], 98),
    )
    ax[0].set_title("Median correlation")
    ax[0].set_xlabel("Ifg number (date)")
    ax[0].set_ylabel("Y or Azimuth (radar coordinates)")
    h = plt.colorbar(im0, ax=ax[0], orientation="horizontal", shrink=0.8)
    ax[1].plot(
        refdate_unique_dt,
        correlation_refdate_stats[:, 3],
        "k-",
        label="max. correlation",
    )
    ax[1].plot(
        refdate_unique_dt,
        correlation_refdate_stats[:, 1],
        "-",
        color="navy",
        label="median correlation",
    )
    ax[1].plot(
        refdate_unique_dt,
        correlation_refdate_stats_top[:, 3],
        "k--",
        label="max. correlation top",
    )
    ax[1].plot(
        refdate_unique_dt,
        correlation_refdate_stats_top[:, 3],
        "k.",
        label="max. correlation bottom",
    )
    ax[1].set_title("Correlation statistics for each reference date")
    ax[1].set_xlabel("Refdate")
    ax[1].set_ylabel("Correlation")
    ax[1].grid()
    ax[1].legend()
    fg.savefig(correlation_ts_fn, dpi=300)
