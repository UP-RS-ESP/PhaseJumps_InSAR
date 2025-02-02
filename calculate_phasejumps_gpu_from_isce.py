#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate absolute phase gradients from unwrapped or wrapped interferograms. Expects ISCE-style files and naming convention. Uses GPU.
Authors: Sofia Viotto, Bodo Bookhagen
bodo.bookhagen@uni-potsdam.de
V0.1 Oct-2024
"""

import os, logging, sys, glob, tqdm, argparse, datetime, warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp

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
            --burst_number 2 \
            --global_stats_csv Olkaria_rg20_az4_global_stats.csv \
            --phasegradient_ts_PNG Olkaria_rg20_az4_phasegradient_ts.png \
            --phasegradient_az_PNG Olkaria_rg20_az4_phasegradient_az.png \
            --summed_phasegradient_az_PNG Olkaria_rg20_az4_summed_phasegradient.png
    
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
        required=False,
    )
    parser.add_argument(
        "--burst_number",
        help='Number of burst overlaps. Can be determined from ESD or other directories. For example, "ls -1 ESD/20240717_20240729/IW3/overlap_??.int | wc -l". Alternatively, you can look at the reference directory.',
        required=True,
    )
    parser.add_argument(
        "--global_stats_csv",
        default="global_phasegrad_coherence_stats.csv",
        help="CSV filename to save global statistics containing dates, phase gradients, and coherence statistics.",
        required=False,
    )
    parser.add_argument(
        "--phasegradient_ts_PNG",
        help="Plot showing the median phase gradient as time series",
        required=False,
    )
    parser.add_argument(
        "--phasegradient_az_PNG",
        help="Plot showing the median phase gradient along azimuzh for all time steps",
        required=False,
    )
    parser.add_argument(
        "--summed_phasegradient_az_PNG",
        help="Plot showing the summed phase gradient statistics along azimuth to identify phase jumps",
        required=False,
    )
    parser.add_argument(
        "--wavelength",
        default=0.05546576,
        help="Radar wavelength for conversion to mm",
        required=False,
    )
    return parser.parse_args()


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
    abs_grad *= 1000 # convert to mm
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
    abs_grad *= 1000 # convert to mm
    abs_grad[cor_gpu < 0.75] = cp.nan
    cor_median = np.round(np.median(cor_gpu).astype(cp.float32).get(), 3)
    cor_mean = np.round(cp.nanmean(cor_gpu).astype(cp.float32).get(), 3)
    cor_std = np.round(cp.nanstd(cor_gpu).astype(cp.float32).get(), 3)
    del cor_gpu
    median_gpu = cp.nanmedian(abs_grad)
    median = np.round(median_gpu.astype(cp.float32).get(), 2)
    mean = np.round(cp.nanmean(abs_grad).astype(cp.float32).get(), 2)
    std = np.round(cp.nanstd(abs_grad).astype(cp.float32).get(), 2)
    p90v = np.round(np.nanpercentile(abs_grad.astype(cp.float32).get(), 90, axis=1), 2)
    medianv = np.round(cp.nanmedian(abs_grad, axis=1).astype(cp.float32).get(), 2)
    meanv = np.round(cp.nanmean(abs_grad, axis=1).astype(cp.float32).get(), 2)
    stdv = np.round(cp.nanmedian(abs_grad, axis=1).astype(cp.float32).get(), 2)

    # calculate severity
    severity = np.round(cp.divide(abs_grad, median_gpu).astype(cp.float32).get(), 0)
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
        p90v,
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


def plot_phasegradient_ts(ph_jump_fn, refdates, phase_grad_ts, cor_ts):
    refList = [datetime.datetime.strptime(date, "%Y%m%d") for date in refdates]
    fg, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(16, 9), dpi=300)
    im0 = ax[0].plot(refList, phase_grad_ts, "+")
    ax[0].set_title("Time series of median phase gradient")
    ax[0].set_xlabel("Date")
    ax[0].set_ylabel("Median phase gradient")

    im1 = ax[1].plot(refList, cor_ts, "+")
    ax[1].set_title("Time series of median coherence")
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("Median coherence")

    fg.tight_layout()
    fg.savefig(ph_jump_fn, dpi=300)


def plot_summed_phasegradient_along_az(
    ph_summed_gradient_fn, abs_grad_median, abs_grad_p90, severity_alongX_proportion
):
    fg, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(16, 9), dpi=300)
    abs_grad_median_sum = np.nansum(abs_grad_median, axis=0)
    abs_grad_median_sum_t = np.nanpercentile(abs_grad_median_sum, 2)
    abs_grad_median_sum[abs_grad_median_sum < abs_grad_median_sum_t] = np.nan
    im0 = ax[0].plot(
        abs_grad_median_sum / np.nanmax(abs_grad_median_sum),
        label="Summed median phase gradient",
    )
    abs_grad_p90_sum = np.nansum(abs_grad_p90, axis=0)
    abs_grad_p90_sum_t = np.nanpercentile(abs_grad_p90_sum, 2)
    abs_grad_p90_sum[abs_grad_p90_sum < abs_grad_p90_sum_t] = np.nan
    im1 = ax[0].plot(
        abs_grad_p90_sum / np.nanmax(abs_grad_p90_sum),
        label="Summed p90 phase gradient",
    )
    severity_alongX_proportion_sum = np.nansum(severity_alongX_proportion, axis=0)
    severity_alongX_proportion_sum_t = np.nanpercentile(
        severity_alongX_proportion_sum, 2
    )
    severity_alongX_proportion_sum[
        severity_alongX_proportion_sum < severity_alongX_proportion_sum_t
    ] = np.nan
    im2 = ax[0].plot(
        severity_alongX_proportion_sum / np.nanmax(severity_alongX_proportion_sum),
        label="Summed severity",
    )
    ax[0].set_title(
        "Summed median and p90 phase gradient along azimuth for all interferograms"
    )
    ax[0].set_xlabel("Azimuth (radar coordinates)")
    ax[0].set_ylabel("Normalized Summed median and p90 abs. phase gradient")
    ax[0].legend()

    severity_alongX_proportion_median = np.nanmedian(severity_alongX_proportion, axis=0)
    severity_alongX_proportion_median_t = np.nanpercentile(
        severity_alongX_proportion_median, 2
    )
    severity_alongX_proportion_median[
        severity_alongX_proportion_median < severity_alongX_proportion_median_t
    ] = np.nan
    im3 = ax[1].plot(severity_alongX_proportion_median)
    ax[1].set_title("Median severity proportion along azimuth for all interferograms")
    ax[1].set_xlabel("Azimuth (radar coordinates)")
    ax[1].set_ylabel("Median severity proportion")

    fg.tight_layout()
    fg.savefig(ph_summed_gradient_fn, dpi=300)


def plot_phasegradient_along_az(
    ph_gradient_fn, abs_grad_median, abs_grad_p90, severity_alongX_proportion
):
    fg, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(16, 9), dpi=300)
    im0 = ax[0].imshow(
        abs_grad_median,
        cmap=plt.cm.viridis,
        vmin=np.nanpercentile(abs_grad_median, 2),
        vmax=np.nanpercentile(abs_grad_median, 98),
    )
    ax[0].set_title("Median phase gradient along azimuth for all interferograms")
    ax[0].set_xlabel("Ifg number")
    ax[0].set_ylabel("Y or Azimuth (radar coordinates)")
    h = plt.colorbar(im0, ax=ax[0], orientation="horizontal")
    h.set_label("Median Phase Gradient [mm]")

    im1 = ax[1].imshow(
        abs_grad_p90,
        cmap=plt.cm.plasma,
        vmin=np.nanpercentile(abs_grad_p90, 2),
        vmax=np.nanpercentile(abs_grad_p90, 98),
    )
    ax[1].set_title("P90 of phase gradient along azimuth for all interferograms")
    ax[1].set_xlabel("Ifg number")
    ax[1].set_ylabel("Y or Azimuth (radar coordinates)")
    h = plt.colorbar(im1, ax=ax[1], orientation="horizontal")
    h.set_label("90th percentile Phase Gradient [mm]")

    im2 = ax[2].imshow(
        severity_alongX_proportion,
        cmap=plt.cm.magma,
        vmin=np.nanpercentile(severity_alongX_proportion, 2),
        vmax=np.nanpercentile(severity_alongX_proportion, 98),
    )
    ax[2].set_title("Severity along X proportion")
    ax[2].set_xlabel("Ifg number")
    ax[2].set_ylabel("Y or Azimuth (radar coordinates)")
    h = plt.colorbar(im2, ax=ax[2], orientation="horizontal")
    h.set_label("Severity proportion [%]")

    fg.tight_layout()
    fg.savefig(ph_gradient_fn, dpi=300)


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    # np.seterr(invalid="ignore")
    warnings.filterwarnings("ignore")
    # could also selectively ignore warnings
    # with warnings.catch_warnings():
    #    warnings.filterwarnings('ignore', r'All-NaN slice encountered')

#    args = cmdLineParser()
    from argparse import RawTextHelpFormatter

    args = argparse.ArgumentParser(
        description=DESCRIPTION, epilog=EXAMPLE, formatter_class=RawTextHelpFormatter
    )
    args.ifg_path = "/raid-gpu2/InSAR/Bolivia/S1_Bermejo/desc10/Bermejo_COP_rg40_az8_noESD/merged/interferograms"
    args.burst_number = 9
    args.global_stats_csv = "Bermejo_rg40_az8_global_stats.csv"
    args.phasegradient_ts_PNG = "Bermejo_rg40_az8_phasegradient_ts.png"
    args.phasegradient_az_PNG = "Bermejo_rg40_az8_phasegradient_az.png"
    args.summed_phasegradient_az_PNG = "Bermejo_rg40_az8_summed_phasegradient.png"
    args.wavelength = 0.05546576
    args.ifg_file = "filt_fine.unw.xml"

    phase2range = float(args.wavelength) / (4.0 * np.pi)
    burst_number = args.burst_number

    logging.info("Getting filelist from %s" % args.ifg_path)
    ifg_date_fn = glob.glob(os.path.join(args.ifg_path, "*", args.ifg_file))
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
        "Creating arrays with %d x %d x %d dimensions for storing phase gradient statistics mean, median, 90p, and std. dev. for each row"
        % (nr_ifg_files, img_length, 4)
    )
    # no need to store all data in large array - this can be calculated for each date separately
    # ds_array = np.empty( (nr_ifg_files, img_length, img_width), dtype=np.float32)
    abs_grad_stats = np.empty((nr_ifg_files, img_length, 4), dtype=np.float32)
    abs_grad_global_stats = np.empty((nr_ifg_files, 3), dtype=np.float32)
    cor_global_stats = np.empty((nr_ifg_files, 3), dtype=np.float32)
    # no need to store full severity matrix in memory - only proportion
    # severity = np.empty( (nr_ifg_files, img_length, 3), dtype=np.float32)
    severity_alongX_proportion = np.empty((nr_ifg_files, img_length), dtype=np.float32)
    # create lists for storing dates
    refdates = []
    secdates = []

    logging.info("Loading phase files and calculating gradients.")
    logging.info(
        "Storing only median, mean, 90percentile, and standard deviation of phase gradient for each row and for each date."
    )
    logging.info(
        "Calculating global statistics from coherence and phase gradient for each time step."
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
            ifg = ifgimg.memMap()[:, 1, :].astype(np.float32)
            cor = corimg.memMap()[:, :, 0].astype(np.float32)
            # ds_array[i,:,:] = data
            (
                abs_grad_stats[i, :, 0],
                abs_grad_stats[i, :, 1],
                abs_grad_stats[i, :, 2],
                abs_grad_stats[i, :, 3],
                abs_grad_global_stats[i, 0],
                abs_grad_global_stats[i, 1],
                abs_grad_global_stats[i, 2],
                cor_global_stats[i, 0],
                cor_global_stats[i, 1],
                cor_global_stats[i, 2],
                severity_alongX_proportion[i, :],
            ) = abs_gradient_severity_float_gpu(ifg, cor, phase2range)
        elif img.dataType == "CFLOAT":
            ifg = np.squeeze(img.memMap()).astype(np.complex64)
            cor = corimg.memMap()[:, :, 0].astype(np.float32)
            (
                abs_grad_stats[i, :, 0],
                abs_grad_stats[i, :, 1],
                abs_grad_stats[i, :, 2],
                abs_grad_stats[i, :, 3],
                abs_grad_global_stats[i, 0],
                abs_grad_global_stats[i, 1],
                abs_grad_global_stats[i, 2],
                cor_global_stats[i, 0],
                cor_global_stats[i, 1],
                cor_global_stats[i, 2],
                severity_alongX_proportion[i, :],
            ) = abs_gradient_severity_complex_gpu(ifg, cor, phase2range)

    global_stats_array = np.c_[abs_grad_global_stats, cor_global_stats]
    save_global_stats2txt(args.global_stats_csv, refdates, secdates, global_stats_array)

    Y_regular_spacing = abs_grad_stats[:,:,0].shape[1]//burst_number
    logging.info("Burst Ovlp areas must be located at ~ %d pixels intervals" % Y_regular_spacing)

    phase_jumps = np.zeros(severity_alongX_proportion.shape, dtype=np.int8)
    phase_jumps_idx = np.where(severity_alongX_proportion > percentage_min)
    phase_jumps[phase_jumps_idx] = 1

    # Findpairs with no phase_jumps (all rows are zero)
    nophase_jumps_idx, = np.where(~np.all(phase_jumps == 0, axis=1) == True)

    # phase_jumps = phase_jumps.dropna(dim="pair", how="all")
    np.all(phase_jumps == 0, axis=0)
    # Drop coordinates without phase_jumps
    phase_jumps = phase_jumps.dropna(dim="Y", how="all")

    date12phase_jumps = list(phase_jumps.pair.values)

    phase_jumps_allDates = []

    if len(args.phasegradient_ts_PNG) > 0:
        logging.info(
            "Creating time series plot of median coherence and phase gradient: %s"
            % args.phasegradient_ts_PNG
        )
        plot_phasegradient_ts(
            args.phasegradient_ts_PNG,
            refdates,
            abs_grad_global_stats[:, 0],
            cor_global_stats[:, 0],
        )

    if len(args.phasegradient_az_PNG) > 0:
        logging.info(
            "Creating maps of median and 90th percentile phase gradient and severity: %s"
            % args.phasegradient_az_PNG
        )
        plot_phasegradient_along_az(
            args.phasegradient_az_PNG,
            abs_grad_stats[:, :, 0],
            abs_grad_stats[:, :, 3],
            severity_alongX_proportion,
        )

    if len(args.summed_phasegradient_az_PNG) > 0:
        logging.info(
            "Creating plot of summed median and 90th percentile phase gradient and severity: %s"
            % args.summed_phasegradient_az_PNG
        )
        plot_summed_phasegradient_along_az(
            args.summed_phasegradient_az_PNG,
            abs_grad_stats[:, :, 0],
            abs_grad_stats[:, :, 3],
            severity_alongX_proportion,
        )
