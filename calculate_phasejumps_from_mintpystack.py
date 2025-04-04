#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------#
# Author Sofia Viotto (viotto1@uni-potsdam.de), Bodo Bookhagen
# (bodo.bookhagen@uni-potsdam.de)
# V0.1 Oct-2024
# V0.2 Jan-2025
# V0.3 Feb-2025

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import copy, logging
import os, glob
import argparse
from tqdm import tqdm
import datetime
import xarray as xr
import matplotlib.pyplot as plt

from mintpy.objects import ifgramStack
from mintpy.utils import readfile
from mintpy.utils import utils1 as ut

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)

import time

start = time.time()

debug = False
# ---------------------------------------#
# Plotting styles
plt.rcParams["font.family"] = "Sans"
plt.rcParams["font.style"] = "normal"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5

# -----------------------------------#

epilog_txt = """
    *****************************************************************
    The number of bursts is listed during ISCE processing. 
    It can be indetified by looking into the reference or secondarys 
    directories of ISCE output. For example:
    ls -1 reference/IW1/*.vrt | wc -l
    ls -1 secondarys/20210412/IW1/burst_0*.vrt | wc -l    

    Note:
    1)  The program requires full extension in the azimuth direction
        (i.e. the stacks must not be subset along the azimuth dimension).
    2)  This program is designed to detect phase jumps within a single sub-swath (IW1, IW2 or IW3).
        It is not compatible yet with stacks created by merging sub-swaths (e.g., IW2-IW3 interferograms).
        If you have a stack with merged subswaths, the option --sub-x can be used to run the program over a specific subswath.
    
    *****************************************************************
    References
    1) Wang et al., 2017 "Improving burst aligment in tops interferometry with BESD",
        10.1109/LGRS.2017.2767575
    2) Zhong et al., 2014 "A Quality-Guided and Local Minimum Discontinuity Based 
        Phase Unwrapping Algorithm for InSAR InSAS Interferograms", 
        10.1109/LGRS.2013.2252880
    ******************************************************************
    """
EXAMPLE = """
    python calculate_phasejumps_from_mintpystack.py --in_dir /path/mintpy  --plot-ind --n-burst 9
    python calculate_phasejumps_from_mintpystack.py --in_dir /path/mintpy  --plot-ind --n-burst 9 --msk-avgCoh --pct 0.10
    python calculate_phasejumps_from_mintpystack.py --in_dir /path/mintpy --pair 20160524_20160711 --n-burst 9  --cmin 0.8
    python calculate_phasejumps_from_mintpystack.py --inDir ./ --n-burst 9
    
    Extract azimuth gradient from unwrapped phase in radar coordinates and find phase jumps.
    Requires an input directory containing the ifgramStack.h5 (usually mintpy/inputs and this 
    code just requires to set the mintpy directory).
    
    The number of burst can be extracted from the ISCE processing directory, 
    for example for IW1 with: ls -1 reference/IW1/*.vrt | wc -l

    The option --plot-ind plots all individual phase jumps as interferogram.
    This is slow, but provides a high level of detail for analysis. Output pngs
    are stored in pj_evaluation/figs and contain the unwrapped phase, 
    absolute phase gradient, and the phase jump analysis. .

    Several output files with statistical information are created in the mintpy
    folder:
    stats_absolute_gradient.txt - File containing summarized statistics (average, median,
        standard deviation) of coherence and absolute phase gradient, as well as 
        the temporal baseline per pair from the stack.

    magnitude_phase_jumps.txt - contains the magnitude of the phase jump for each interferogram pair.
    
    exclude_listdate12_interferograms_by_phase_jump.txt - summary of pairs and
        indices used for the phase-jump assessment
    
    exclude_dates_by_phase_jumps.txt - list of dates where more than 50% of
        interferograms contained phase jump. These dates may be removed from
        processing.

"""

parser = argparse.ArgumentParser(
    description=EXAMPLE,
    epilog=epilog_txt,
    usage=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--inDir",
    "-i",
    dest="in_dir",
    help="Input folder containing input/ifgramStack.h5. This is usually just the mintpy folder.",
    default=os.getcwd(),
    required=True,
)

parser.add_argument(
    "--n-burst",
    "-n",
    dest="n_burst",
    type=int,
    help="Number of bursts processed",
    required=True,
)
parser.add_argument(
    "--pair", default=None, help="Perfom calculations only for specific pair (YYYYMMDD_YYYYMMDD).", dest="pair"
)
# parser.add_argument(
#     "--pmin",
#     default=80.0,
#     help="Minimun percentage of pixels jumping along a row to be defined as phase jump [default: %(default)s]",
#     dest="min_pct",
#     type=float
# )

parser.add_argument(
    "--cmin",
    default=0.75,
    help="Minimum coherence to mask out pixels [default: %(default)s]. This is used to mask pixels from the unwrapped phase and gradient along the azimuth direction. It is furthermore used as a threshold for the mean coherence mask. You can increase the threshold if you have an area with high coherence (such as the dry Central Andes), and you may need to lower this in areas with low coherence to obtain a sufficient pixel number per row to detect phase jumps.",
    dest="cmin",
    type=float,
)

parser.add_argument(
    "--pct",
    default=0.25,
    help="Percentile of the distribution of coherence pixel number per row, to define reliable rows. Range value (0.0-1.0) [default: %(default)s]. We don't recommend using values higher than 0.5, because this will use either 50% of the width or either the 50th percentile of the distribution of good pixels per row. Higher numbers will lower the number of pixels that can be used for detecting phase jumps. A values of 0.25 works in most cases.",
    dest="pct_row",
    type=float,
)

parser.add_argument(
    "--msk-avgCoh",
    default=False,
    help="Compare the number of pixels per row and pair with an average coherence mask (mask defined according to --cmin), in order to improve the phase jump detection at the burst overlaps.",
    dest="mask_coh",
    action="store_true",
)

parser.add_argument(
    "--pj-thr",
    default=5.0,
    dest="pj_thr_mm",
    type=float,
    help="Threshold in mm of maximum accumulated phase jump. [default: %(default)s].",
)

parser.add_argument(
    "--sub-x",
    default=None,
    help="Define the area of interest along x.",
    dest="subX",
    nargs=2,
    type=int,
)

parser.add_argument(
    "--plot-ind",
    dest="plot_ind",
    help="Plot individual phase jumps. This is slow, but provides a great level of detail for further analysis.",
    action="store_true",
)

args = parser.parse_args()
inps = args.__dict__


def calculate_stats_arrays(ds_unw, ds_coh):
    stats_abs_grad = np.empty((3), dtype=np.float32)
    stats_coh = np.empty((3), dtype=np.float32)

    stats_abs_grad[0] = np.round(np.nanmedian(ds_unw), 2)
    stats_abs_grad[1] = np.round(np.nanmean(ds_unw), 2)
    stats_abs_grad[2] = np.round(np.nanstd(ds_unw), 2)

    stats_coh[0] = np.round(np.nanmedian(ds_coh), 2)
    stats_coh[1] = np.round(np.nanmean(ds_coh), 2)
    stats_coh[2] = np.round(np.nanstd(ds_coh), 2)

    return stats_abs_grad, stats_coh


# -------Plot functions
def plot_ind(arr_unw, arr_abs_grad, sev_pct, min_pct, fn_out, date12, orbit):
    fig_size = (10, 7)

    title = "Pair %s (Mask Coherence)" % date12

    # Plot
    fig, axs = plt.subplots(
        ncols=3,
        figsize=fig_size,
        sharex=False,
        sharey=True,
        gridspec_kw={"width_ratios": [3, 3, 1]},
    )
    fig.subplots_adjust(top=0.8)
    # Title for the entire figure
    fig.suptitle(title, fontsize=11, fontweight="bold")

    #    Subplot 1: Unwrapped Phase
    axs[0].set_title(r"Unwrap Phase $\varphi$", fontsize=11)
    unwPlot = axs[0].imshow(
        arr_unw,
        cmap="RdBu",
        interpolation="nearest",
        vmin=np.nanpercentile(arr_unw, 10),
        vmax=np.nanpercentile(arr_unw, 90),
        aspect="auto",
    )
    fig.colorbar(
        unwPlot,
        label=r"$\varphi$ [Rad]",
        ax=axs[0],
        pad=0.03,
        shrink=0.6,
        aspect=30,
        orientation="vertical",
    )
    axs[0].set_ylabel("Azimuth")
    axs[0].set_xlabel("Range")
    axs[1].set_xlabel("Range")

    # Subplot 2:
    axs[1].set_title(r"Absolute ($\varphi_{(i+1,j)}-\varphi_{(i,j)}$)", fontsize=11)
    vmax = 1.5  # np.nanpercentile(arr_abs_grad, 98)
    vmin = np.nanpercentile(arr_abs_grad, 2)
    gradPlot = axs[1].imshow(
        arr_abs_grad,
        cmap="viridis",
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    fig.colorbar(
        gradPlot,
        ax=axs[1],
        label=r"$|\delta\varphi_{az}|$ [mm]",
        pad=0.03,
        shrink=0.6,
        aspect=30,
        orientation="vertical",
    )

    # Subplot 3: Jumps Proportion
    axs[2].set_title(r"$\Sigma$ $Jumps_{(i)}$", fontsize=11)
    axs[2].plot(
        sev_pct,
        np.arange(0, sev_pct.shape[0], 1),
        lw=0.5,
        c="black",
    )
    axs[2].set_xlabel("Jumps [%Pixels]")
    axs[2].set_xticks([0, 50, 100])
    axs[2].set_xticklabels([0, 50, 100])
    axs[2].axvline(x=min_pct, c="r", lw=0.5, zorder=0)

    # Axis inversion based on orbit
    if orbit.lower().startswith("a"):
        axs[0].invert_yaxis()

    elif orbit.lower().startswith("d"):
        axs[0].invert_xaxis()
        axs[1].invert_xaxis()
        # Set aspect ratio to keep all plots the same height
    for ax in axs[:2]:
        ax.set_aspect("auto")
        axs[2].set_aspect(aspect="auto", adjustable="box")
    # Adjust layout for better spacing
    fig.subplots_adjust(wspace=0.3)

    fig.savefig(fn_out, dpi=150)

    plt.clf()
    plt.close()


# #----------------Aux fuctions
# def group_coord_by_distance(y, fr, ovlp_reg):

#     y_gps = np.split(y, np.where(np.diff(y) > ovlp_reg/2)[0] + 1)
#     fr_gps = np.split(fr, np.where(np.diff(y) > ovlp_reg/2)[0] + 1)
#     del y,fr

#     y_final = []

#     for gp, fr_gp in zip(y_gps, fr_gps):
#         if gp.shape[0] > 1:
#             try:
#                 # #Find the coordinate that have the max frequency
#                 y_final.append(copy.deepcopy(gp[fr_gp == np.max(fr_gp)].item()))

#             except:
#                 #If the frequency is the same append the list
#                 y_final.extend(copy.deepcopy(gp))
#         elif gp.shape[0]==1:
#             y_final.append(gp[0])
#         else:
#             continue
#     return y_final


# ------------------------------------#
# -Fuctions added on Feb/20/2025
def modify_network(da, date12List_Keep, date12List_Skip, fn_report=None, clean=False):
    da1_Keep, da2_Skip = da.copy(), da.copy()

    da1_Keep.loc[{"pair": date12List_Skip}], da2_Skip = np.nan, da2_Skip.sel(
        pair=date12List_Skip
    )

    # Report
    if fn_report != None:
        clean = True
        with open(fn_report, "w") as fl:
            for item in date12List_Skip:
                fl.write(item + "\n")
            fl.close()
            # Esto tiene que ir a otra funciont

    if clean == True:
        fn_in = os.path.join(os.path.dirname(fn_report), "stats_absolute_gradient.txt")
        fn_out = fn_in.replace(".txt", "_skipped_ifgs.txt")
        with open(fn_in, "r") as fl:
            lines = fl.readlines()
            fl.close()
        ll = []
        with open(fn_in, "w") as fl:
            for item in lines:
                if item.split(",")[0] in date12List_Skip:
                    ll.append(item)
                else:
                    fl.write(item)
            fl.close()
        #
        with open(fn_out, "w") as fl:
            for item in lines[:11]:
                fl.write(item)
            for item in ll:
                fl.write(item)
            fl.close()

    return da1_Keep, da2_Skip


def measure_pj(da, yList, n_burst):
    # Two ways to measure phase jumps
    # As we need a unique value to compare phase jumps
    da2 = da.copy()

    # Select only phase jump coordinates
    da2 = da2.sel(Y=yList)

    # Remove pairs with no data to evaluate at those positions
    da2 = da2.dropna(dim="pair", how="all")

    # From phase jump coordinates,
    # keep only common coordinates among the pairs
    da3 = da2.copy()
    da3 = da3.dropna(dim="Y", how="any")

    # N overlapping areas
    n_ovlp = n_burst - 1
    if (da3.Y.shape[0] == 0) or (da3.Y.shape[0] < n_ovlp):
        # No coordinates left
        # Then use the mean value of the coordinates

        da_out = da2.mean(dim="Y")
        da_out *= n_ovlp

    else:

        # Calculate the sum
        da_out = da3.sum(dim="Y")

    if debug == True:
        plt.figure()
        plt.hist(da_out.data, bins=10)
        plt.title("Phase Jump Accumulated")
    return da_out


# --------------------------------#
# Report
# ---------------------------------#


def ifg2idx(List1, List2):
    idx = [List1.index(i) for i in List2]
    return list(np.asarray(idx, dtype=str))


def report_ifgs2drop(date12List, date12List_Skip, date12List_Drop, out_fn):
    # Get index from those dates
    idxDate12List_Skip = ifg2idx(List1=date12List, List2=date12List_Skip)
    idxDate12List_Drop = ifg2idx(List1=date12List, List2=date12List_Drop)

    # Report list of interferograms to be excluded
    with open(out_fn, "w") as fl:
        # Report dates skipped
        fl.write("*" * 10 + "\n\n")
        fl.write("List of pairs skipped from assessment (mean coherence < 0.4): \n")
        fl.write(",".join(date12List_Skip) + "\n")
        fl.write("\nIndexes pairs skipped:\n")
        fl.write(",".join(idxDate12List_Skip) + "\n")
        fl.write("\nTotal pairs skipped: %s \n\n" % str(len(date12List_Skip)))
        fl.write("*" * 10 + "\n\n")
        fl.write("\nList of pairs with phase jumps larger than threshold: \n")
        fl.write(",".join(date12List_Drop) + "\n")
        fl.write("\nIndices of pairs with phase jump larger than threshold:\n")
        fl.write(",".join(idxDate12List_Drop) + "\n")
        fl.write(
            "\nTotal pairs with phase jump larger than threshold: %s \n\n"
            % str(len(date12List_Drop))
        )
        fl.close()


def report_dates2drop(date12List_Keep, date12List_Drop, out_fn):
    # Count ifgs  per date
    datesList = [pair.split("_")[0] for pair in date12List_Keep]
    datesList.extend([pair.split("_")[1] for pair in date12List_Keep])
    #
    dates, fr_ifgs = np.unique(datesList, return_counts=True)

    # Count ifgs with phase jump
    datesList_pj = [i.split("_")[0] for i in date12List_Drop]
    datesList_pj.extend([i.split("_")[1] for i in date12List_Drop])

    #
    dates_pj, freq_ifgs_pj = np.unique(datesList_pj, return_counts=True)

    # Percetange of ifgs with phase jump
    pct_ifgs_pjump = np.asarray(
        [
            n_ifgs_phase_jump * 100 / fr_ifgs[dates == date_i]
            for date_i, n_ifgs_phase_jump in zip(dates_pj, freq_ifgs_pj)
        ]
    ).flatten()

    datesList_Drop = list(dates_pj[pct_ifgs_pjump > 50])
    # logging.info(datesList_Drop)
    with open(out_fn, "w") as fl:
        fl.write("Dates with more than 50% of the pairs with significant phase jump \n")
        fl.write(",".join(datesList_Drop))
        fl.close()


def report_pj(da_pj, date12List_Keep, out_fn_report):
    # Header report
    header = (
        "# Pairs found with systematic phase jumps \n"
        "# Magnitude_PJ_mm: Magnitude of the phase jump in mm, calculated with: \n"
        "# mean(phase jump of overlapping areas) * number of overlapping areas \n"
    )

    # Report
    report_txt = []
    for idx, date12 in enumerate(date12List_Keep):
        mag_pj = da_pj.sel(pair=date12).item()
        report_txt.extend([date12 + "\t" + str(np.round(mag_pj, 2))])
        del mag_pj

    # Report
    if len(report_txt) > 0:
        # -Report each pair, number of phase jumps and coordinates
        with open(out_fn_report, "w") as fl:
            fl.write(header)
            fl.write("# DATE12 \t\tMagnitude_PJ_mm\n")

            for item in report_txt:
                fl.write("%s\n" % item)
            fl.close()


def report_stats(inps):

    refList = [
        datetime.datetime.strptime(pair.split("_")[0], "%Y%m%d")
        for pair in inps["date12List"]
    ]
    secList = [
        datetime.datetime.strptime(pair.split("_")[1], "%Y%m%d")
        for pair in inps["date12List"]
    ]
    Bt = [(sec - ref).days for ref, sec in zip(refList, secList)]
    stats_coh = np.asarray(inps["stats_coh"])  # [:,:-1]
    stats_abs_grad = np.asarray(inps["stats_abs_grad"])
    size = 1 + stats_coh.shape[1] + stats_abs_grad.shape[1]

    array = np.empty((len(refList), size), dtype=float)
    array[:, 0] = Bt
    array[:, 1 : stats_coh.shape[1] + 1] = stats_coh
    array[:, stats_coh.shape[1] + 1 :] = stats_abs_grad

    # Save Txt file with stats from Azimuth Gradient
    outFile = os.path.join(inps["out_dir"], "stats_absolute_gradient.txt")

    header = (
        "# No Data Values (Zero Values) were excluded from calculations.\n"
        "# Coherence statistics were derived from all non-NaN pixels.\n"
        "# Pixels with coherence < 0.75 were masked out during the calculation of absolute azimuth gradient and corresponding severity.\n"
        "# The number of masked-out pixels varies between pairs.\n"
        "## Column Names/Prefixes:\n"
        "# Btemp: Temporal Baseline\n"
        "# Coh: Coherence\n"
        "# Grad: Absolute gradient along the azimuth direction \n"
        "# Med: Median, Std: Standard deviation\n\n"
    )

    stats_name = ["Grad_Median_mm", "Grad_Mean_mm", "Grad_Std_mm"]
    # Prepare name of columns
    colCoh = copy.deepcopy(stats_name)
    colCoh = [i.replace("Grad", "Coh").replace("_mm", "") for i in colCoh]
    colCoh = ",".join(colCoh)
    colabs_grad = stats_name
    colabs_grad = ",".join(colabs_grad)
    columns = "DATE12,Btemp[days]," + colCoh + "," + colabs_grad + "\n"

    with open(outFile, "w") as fl:
        fl.write(header)
        fl.write(columns)
        for line, pair in zip(array, inps["date12List"]):
            line = list(np.round(line, 2).astype(str))
            line = pair + "," + ",".join(line) + "\n"
            fl.write(line)


# -----------------------------#
# Mask
def remove_trend(da, n_burst, length, mode="normalize", step=None):
    if mode == "normalize":
        da_median = da.median(dim="pair")
        da /= da_median

    elif mode == "smooth":
        if step == None:
            step = length // (n_burst * 3)
        # Lineal + regional trend
        trend = da.coarsen(Y=step, boundary="trim").median()
        trend_interp = trend.interp(Y=da.Y.values, method="linear")
        da -= trend_interp

    return da


def mask_by_avgCoh(inps, da_sev_pct, da_CohPx_cts):

    # calculate average coherence WITHOUT excluding pairs
    avgCoh = ut.temporal_average(
        inps["fn_stack"], datasetName="coherence", outFile=False
    )[0]

    maskCoh = np.zeros(avgCoh.shape, dtype="int")

    maskCoh = np.where(avgCoh >= inps["cmin"], 1, 0)

    arr_maskCoh_sum = np.nansum(maskCoh, axis=1)

    del avgCoh, maskCoh

    # Compare ammount of coh pixels per row to a mask of coherence
    da_sev_pct = da_sev_pct.where(da_CohPx_cts >= arr_maskCoh_sum)
    return da_sev_pct


def dynamic_threshold2mask(inps, da_CohPx_cts, da_sev_pct):
    # ----------
    t_CohPx = np.nanpercentile(da_CohPx_cts, inps["pct_row"] * 100)
    t_width = inps["width"] * inps["pct_row"]
    # ----------
    if t_width > t_CohPx:
        da_sev_pct.data = np.where(da_CohPx_cts.data > t_width, da_sev_pct.data, np.nan)

        # Activate mask  based on coherence to further mask out rows
        logging.info(
            "Mask based on average coherence, with %s as minimum coherence"
            % np.round(inps["cmin"], 2)
        )
        inps["mask_coh"] = True

    else:
        da_sev_pct = da_sev_pct.where(da_CohPx_cts > t_CohPx)

    return inps, da_sev_pct


# ------------------------------------------
# Phase Jump detection


def refine_detection(da_med_abs_grad, yList, ovlp_reg, n_burst):
    # Total of pairs
    n_pairs = da_med_abs_grad.pair.shape[0]
    # Group coordinates
    y, fr = np.unique(yList, return_counts=True)

    # Filter out coordinates
    # based on frequency
    t_pairs = 0.01 * n_pairs
    # Apply filter
    y, fr = y[fr > t_pairs], fr[fr > t_pairs]

    # --------------------
    # Search area
    half_ovlp = ovlp_reg // 2

    yList_out = []

    for n in range(1, n_burst):
        # Group coordinates by range
        y_inf, y_sup = (n * ovlp_reg - half_ovlp), (n * ovlp_reg + half_ovlp)
        y_sub, fr_sub = y[(y >= y_inf) & (y < y_sup)], fr[(y >= y_inf) & (y < y_sup)]

        try:
            y_ = y_sub[fr_sub == np.nanmax(fr_sub)]
            if y_.shape[0] > 1:
                # In case of equal frequency, keep the coordinate
                # with the maximum magnitude
                med_mag = da_med_abs_grad.loc[{"Y": y_}].median(dim="pair").data
                y_max = y_[med_mag == np.max(med_mag)]
            else:
                y_max = y_
            yList_out.extend(list(y_max))
        except:
            pass

    return yList_out


def coarse_detection_pj(da, n_burst, length):

    # Number of pairs
    n_pairs = da.pair.shape[0]

    # Detrend
    da_detrended = remove_trend(da, n_burst, length)

    # Calculate gradient of the severity
    arr_grad = np.diff(da_detrended.data, axis=1, prepend=0)

    # Obtain std
    std_pair = np.nanstd(arr_grad, axis=1)

    # -------
    yList = []

    # Keep coordinates where peak of phase jump > 3* std
    for i in range(0, n_pairs):
        y_temp = list(np.where(arr_grad[i, :] > (3 * std_pair[i]))[0])
        if len(y_temp) > 0:
            yList.extend(y_temp)

    return yList


def detect_pj_coordinates(inps, da_sev_pct, da_med_abs_grad, da_CohPx_cts):
    """
    Phase jump detection based on peak detection (coarse detection step)
    followed by frequency and magnitude refinement (refine detection step)

    """
    n_burst = inps["n_burst"]
    length = inps["length"]
    ovlp_reg = length // n_burst

    def steps():
        # 1) Find coordinates with phase jump
        yList = coarse_detection_pj(da=da_sev_pct, n_burst=n_burst, length=length)

        # 2) Refine coordinate position based on frequency & magnitude
        yList = refine_detection(da_med_abs_grad, yList, ovlp_reg, n_burst)

        return yList

    def check_result():
        # 3) If coordinates fails by giving more coordinates than
        # burst_overlapping areas that means the area has low coherence
        # then repeat 1-2
        # As well, if there area coordinates too close with the same frequency
        dy = np.round(np.diff(yList) / ovlp_reg, 2)
        ff = np.floor(dy)
        dy -= ff
        rerun = False
        print(dy)
        if (len(yList) > (n_burst - 1)) or (np.min(dy) < 0.75):
            logging.warning(
                "Inconsistency in detected coordinates! "
                "There are more coordinates than burst overlapping areas."
                "or there is not regular scpacing between coordinates. "
            )
            logging.info(
                "Improving detection by increasing minimun coherence"
                "and re-masking severity file."
            )

            rerun = True

            if debug == True:
                print(yList)

        return rerun

    # -
    d_coh = (0.95 - inps["cmin"]) / 5
    # -
    cmin = copy.deepcopy(inps["cmin"])

    # -
    for i in range(0, 5):
        yList = steps()
        rerun = check_result()
        if rerun:
            #
            inps["cmin"] = cmin + (d_coh * i)
            #
            da_sev_pct = mask_by_avgCoh(inps, da_sev_pct, da_CohPx_cts)
        else:
            break

    return da_sev_pct, yList, inps


# ------------------------------------#


def analyze_phase_jump(inps, da_sev_pct, da_CohPx_cts, da_med_abs_grad):
    """
    Reads severity (sev) files and identifies phase phase jumps at regular intervals.
    The coordinates of phase jumps are reported.

    Parameters:
    -----------
    da_sev_pct: data array (pair, azimuth) ; percentage of pixels jumping by row ,
    da_CohPx_cts: data array (pair, azimuth); Number of data pixels (no Nan) by row,
    da_med_abs_grad: data array (pair,azimuth);  median absolute gradient by row,

    inps : dict
        A dictionary containing the following keys:
        - 'date12List'  : list; List of interferograms. Format 'YYYYMMDD_yyyymmdd'
        - 'min_pct'     : float; Minimum  percentage to consider a peak of detected pixels as a phase jump.
        - 'in_dir'      : str; Input directory
        - 'n_burst'     : str; Number of bursts expected.
        - 'min_pct'     : float; Minimun percentage of pixels jumping along the row.
        - 'out_dir'     : str; Output directory.
        - 'length'      : int; file length, i.e. number of rows along azimuth
        - 'n_burst'     : int; Number of bursts in the dataset.

    """
    # ------------------------------
    # Input
    date12List = inps["date12List"]
    date12List_Skip = inps["date12List_Skip"]
    date12List_Keep = sorted(list(set(date12List) - set(date12List_Skip)))
    n_burst = inps["n_burst"]
    # min_pct = inps["min_pct"]
    # length=inps['length']
    # ovlp_reg = length // n_burst

    # ------------------------------
    # Output reports
    out_fn_report = os.path.join(inps["in_dir"], "magnitude_phase_jumps.txt")
    out_fn_excList_ifg = os.path.join(
        inps["in_dir"], "exclude_listdate12_interferograms_by_phase_jump.txt"
    )
    out_fn_excList_dat = os.path.join(
        inps["in_dir"], "exclude_dates_by_phase_jumps.txt"
    )

    # ----------------
    # STAGE 1: Prepare Data
    # --------------
    # Determine which pairs and rows can be used to
    # find the coordinates of the burst
    # overlapping areas
    # Only high-coherence rows and pairs are used to
    # find the coordinates of the phase jump
    # --------------

    # 1) Exclude from assessment pairs with mean coherence < 0.4
    if len(date12List_Skip) > 0:
        logging.warning(
            "Excluding a total of {} pairs from the assessment due to mean coherence < 0.4".format(
                len(date12List_Skip)
            )
        )
        # -
        da_CohPx_cts, _ = modify_network(
            da_CohPx_cts,
            date12List_Keep=date12List_Keep,
            date12List_Skip=date12List_Skip,
        )
        da_med_abs_grad, _ = modify_network(
            da_med_abs_grad,
            date12List_Keep=date12List_Keep,
            date12List_Skip=date12List_Skip,
        )
        da_sev_pct, _ = modify_network(
            da_sev_pct, date12List_Keep=date12List_Keep, date12List_Skip=date12List_Skip
        )

    # No data
    da_CohPx_cts = da_CohPx_cts.where(da_CohPx_cts != -999)
    da_CohPx_cts = da_CohPx_cts.where(da_CohPx_cts != 0)

    # Mask out rows based on number of pixels per row, per pair
    inps, da_sev_pct = dynamic_threshold2mask(
        inps, da_CohPx_cts=da_CohPx_cts, da_sev_pct=da_sev_pct
    )

    # Apply an extra mask of coherence
    if inps["mask_coh"] == True:
        da_sev_pct = mask_by_avgCoh(
            inps, da_sev_pct=da_sev_pct, da_CohPx_cts=da_CohPx_cts
        )

    # ------------------
    # STAGE 2:
    # Phase jump coordinates detection
    # ------------------

    da_sev_pct, yList, inps = detect_pj_coordinates(
        inps, da_sev_pct, da_med_abs_grad, da_CohPx_cts
    )

    if debug == True:
        logging.info("Coordinates of phase jumps %s." % yList)

    # -------------------------------
    # Detection based on height of the peak
    # y_iter=[]
    # d_pct=1
    # n_iter=int((100-min_pct)//d_pct)

    # with tqdm(total=n_iter, desc="Estimating Phase Jumps Coordinates",ncols=100, bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
    #     for n in range(0,n_iter):
    #         min_pct_1=min_pct+n*d_pct
    #         y_iter.extend(get_phase_jump_cord(da_sev_pct,min_pct_1,ovlp_reg,n_burst))
    #         pbar.update(1)

    # y,fr=np.unique(y_iter, return_counts=True)
    # if debug==True:
    #     print(y,fr)
    # yList=group_coord_by_distance(y, fr, ovlp_reg)
    # ------------------------------------

    # --------------#
    # STAGE 3
    # Estimate phase jump magnitude & report
    # ------------- #
    # Estimate phase jumps magnitude
    da_pj = measure_pj(da_med_abs_grad, yList=yList, n_burst=n_burst)

    # Update list of dates12
    date12List_Keep = sorted(list(da_pj.pair.values))
    date12List_Skip = sorted(list(set(date12List) - set(date12List_Keep)))

    # Report estimated phase jumps magnitude FROM ALL
    # dates assessed
    report_pj(da_pj, date12List_Keep, out_fn_report)

    # ----------------------------------------#
    # if debug==True:
    #     #Add the median phase jump per coordinate
    #     values = []
    #     for pair in date12List_Keep:
    #         arr = da_med_abs_grad.sel(pair=pair).copy()
    #         arr = arr.sel(Y=np.asarray(yList))
    #         arr = arr.data.flatten()
    #         arr = np.round(arr, 2)
    #         arr = list(arr)
    #         arr = [str(i) for i in arr]

    #         values.append("," + ",".join(arr) + "\n")
    #         del arr,pair

    #     # Open stats and add the new stats:
    #     logging.info("Median Absolute Gradient per Coordinate saved at %s " % out_fn_stats)

    #     with open(out_fn_stats, "r") as fl:
    #         lines = fl.readlines()
    #         fl.close()

    #     y_txt = ["Grad_Median_Y-" + str(int(i)) + "[mm]" for i in yList]
    #     lines[10] = lines[10].replace("\n", "," + ",".join(y_txt) + "\n")

    #     # Add the statis
    #     subset_lines = copy.deepcopy(lines[11:])
    #     subset_lines = [i.replace("\n", j) for i, j in zip(subset_lines, values)]
    #     lines[11:] = subset_lines
    #     with open(out_fn_stats, "w") as fl:
    #         for line in lines:
    #             fl.write(line)
    #         fl.close()
    #     del subset_lines,y_txt,lines,values
    # ----------------------------------------------

    # -------------------------
    # STAGE 4:
    # Determine pairs/dates to be dropped from
    # -----------------

    # Threshold
    pj_thr = inps["pj_thr_mm"]

    # Mask out & remove pairs with phase jumps below threshold
    da_pj = da_pj.where(da_pj >= pj_thr)
    da_pj = da_pj.dropna(dim="pair")

    # Update list of dates12
    date12List_Drop = sorted(list(da_pj.pair.values))

    report_ifgs2drop(
        date12List=date12List,
        # - dates12List_Skip : from the assessment
        date12List_Skip=date12List_Skip,
        # - dates12List_Drop : magnitude of the phase jump > threshold
        date12List_Drop=date12List_Drop,
        out_fn=out_fn_excList_ifg,
    )

    # -----------------------------------------#
    # #Find the pairs with median ds abs grad larger 1 mm
    # med_threshold = 1.0
    # da_med_abs_grad = da_med_abs_grad.where(
    #     da_med_abs_grad >= med_threshold
    # )
    # da_med_abs_grad = da_med_abs_grad.dropna(dim="pair", how="all")

    # #Keep the dates
    # date12List_Drop = list(da_med_abs_grad.pair.values)

    # logging.info("Phase jump is significant if the median jump of the row is >%s mm\n"
    # % np.round(med_threshold, 2)+
    #     " Total of pairs found with significant phase jumps: %s"
    #     % len(date12List_Drop)
    # )

    # --------------------------------------------------#
    # y_summary = []
    # report_txt = []
    # y_pj_pair = []
    # logging.info(
    #     "Reporting phase jumps %s"
    #     % out_fn_report
    # )
    # for idx, date12 in enumerate(date12List_Drop):

    #     da_med_abs_grad_pair = da_med_abs_grad.sel(pair=date12).copy()
    #     y_pj_pair = list(da_med_abs_grad_pair.dropna(dim="Y").Y.values)

    #     if len(y_pj_pair) > 0:
    #         # Store
    #         y_summary.extend(copy.deepcopy(y_pj_pair))

    #         #Transform to string for reporting
    #         y_pj_pair = [str(i) for i in y_pj_pair]
    #         spacing = "\t" * ((n_burst // 2 - len(y_pj_pair) // 2))

    #         # Calculate the average phase jump
    #         avg_pj = np.nanmean(da_med_abs_grad_pair.data)

    #         report_txt.extend(
    #             [
    #                 date12
    #                 + "\t"
    #                 + str(len(y_pj_pair))
    #                 + "\t\t"
    #                 + ",".join(y_pj_pair)
    #                 + spacing
    #                 + str(np.round(avg_pj, 2))
    #             ]
    #         )
    #     else:
    #         # Remove date12 if there is not date to report
    #         date12List_Drop = [
    #             i for i in date12List_Drop if i != date12
    #         ]
    #         continue
    #     del da_med_abs_grad_pair
    # ----------------------------------------------

    # ------------------------------
    # # Provide a summary of the coordinates found
    # y_, cts_ = np.unique(y_summary, return_counts=True)
    # with open(out_fn_summary, "w") as fl:
    #     fl.write("#Az_cord\tCounts\n")
    #     for y_i, cnt_i in zip(y_, cts_):
    #         fl.write("%s\t%d\n" % (y_i, cnt_i))

    # del y_, cts_
    # --------------------------

    # --------------------------
    # Report dates
    # --------------------------
    report_dates2drop(
        date12List_Keep=date12List_Keep,
        date12List_Drop=date12List_Drop,
        out_fn=out_fn_excList_dat,
    )


def unwrap_phase2azimuth_gradient(inps):
    # ----------------
    # Input
    # ----------------
    fn_stack = inps["fn_stack"]

    # Y=length=azimuth coordinate
    # X=width=range coordinates
    length = inps["length"]
    # width=inps['width']
    date12List = inps["date12List"]
    wavelength = inps["wavelength"]
    phase2range = wavelength / (4.0 * np.pi)
    # min_pct=inps['min_pct']
    min_coh = inps["cmin"]
    # ----------------
    # Output
    # -----------------
    # 3D (time, azimuth, range)
    # fn_abs_grad=inps['fn_abs_grad_2D']
    # fn_sev = inps['fn_sev']

    # 2D (time, azimuth)
    fn_pxCoh_cts = inps["fn_pxCoh_cts"]
    fn_sev_pct = inps["fn_sev_pct"]
    fn_med_abs_grad = inps["fn_med_abs_grad"]
    # fn_treshold=inps['fn_tre']

    # -------------Begining Main Operations-------------#

    # --------------------
    # Calculate Azimuth Gradient
    # -----------------
    # Calculate the diff along the azimuth direction to later detect phase jumps
    # v(m,n)  from the equation of Zhong et al., 2014:
    # v(m,n) = Int((φ m,n − φ m−1,n )/2π) (Eq 2, page 216)
    # ds_grad_az= (φ m,n − φ m−1,n )
    # axis 0 is equal to Y direction=azimuth direction for 2D arrays
    # ds_grad_az=np.diff(ds_unw, axis=0,prepend=0)

    # Create containers for arrays
    arr_med_sev_acrX = np.zeros((len(date12List), length), dtype=float)
    arr_sev_y = np.zeros((len(date12List), length), dtype=float)
    arr_threshold = np.zeros(len(date12List), dtype=float)
    arr_NoNans = np.zeros((len(date12List), length), dtype=int)
    arr_NoNans -= 999
    # Create containers for stats
    stats_abs_grad = np.zeros((len(date12List), 3), dtype=float)
    stats_coh = np.zeros((len(date12List), 3), dtype=float)
    date12List_Skip = []

    logging.info("Calculating Absolute Gradient in the Azimuth Direction.")

    with tqdm(
        total=len(date12List),
        desc="Pairs processed",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        # Loop to generate severity
        for idx, pair in enumerate(date12List):

            # ---------------------------------#
            # Note: Loading the entire dataset is not a good idea for large files.
            # The memory might not be sufficient. It is better and faster
            # to apply the function one pair at a time.
            # ---------------------------------#
            # Read data
            if inps["subX"] != None:
                # (x0, y0, x1, y1)
                x0, x1 = np.min(inps["subX"]), np.max(inps["subX"])
                bbox = [x0, 0, x1, length]

            else:
                bbox = None

            arr_unw = readfile.read(
                fn_stack, datasetName="unwrapPhase-" + pair, box=bbox
            )[0]
            arr_coh = readfile.read(
                fn_stack, datasetName="coherence-" + pair, box=bbox
            )[0]
            # Apply mask
            arr_unw[arr_coh < min_coh] = np.nan

            # Remove no data
            arr_coh[arr_coh == 0] = np.nan
            arr_unw[arr_unw == 0] = np.nan

            # Calculate absolute gradient
            # Do not prepend a zero, as the first line is already zero
            arr_abs_grad = np.zeros(arr_unw.shape, dtype=float)
            arr_abs_grad[1:, :] = np.abs(np.diff(arr_unw, axis=0))

            #  Set the last lines to zero, to avoid border effects
            arr_abs_grad[:2, :] = np.nan
            arr_abs_grad[-2:, :] = np.nan

            # Convert to displacement in milimeters
            arr_abs_grad *= phase2range
            arr_abs_grad *= 1000
            # Set outliers to nan as well, BEFORE deriving statistics
            p99 = np.nanpercentile(arr_abs_grad, 99)

            arr_abs_grad[arr_abs_grad > p99] = np.nan
            del p99

            # Obtain stats
            stats_abs_grad[idx, :], stats_coh[idx, :] = calculate_stats_arrays(
                arr_abs_grad, arr_coh
            )

            # ------------------------------
            # Skip pairs based on coherence
            if stats_coh[idx, 1] < 0.4:
                date12List_Skip.append(pair)

            # -----------------------------------------
            # Calculate the sev (magnitude) of the phase jump
            # -----------------------------------------
            # the threshold is the median of the absolute gradient
            threshold = stats_abs_grad[idx, 0]

            # -----------------------
            ### This option produces a noiser mask
            # sev = np.round(np.divide(arr_abs_grad, threshold), 0)
            # sev[sev > 1] = 1
            # -----------------------

            # Dims (row-1,col)=(azimuth,range)=(y,x)
            sev = np.zeros(arr_abs_grad.shape, dtype=float)
            sev = np.where(arr_abs_grad >= threshold, 1, sev)
            mask = np.isnan(arr_abs_grad)
            sev[mask] = np.nan

            # Dims (row-1)
            sev_acrX = np.nansum(sev, axis=1)

            # Count NoNans pixels to 1) compute percentage along the row, and
            # 2) determine if the row is reliable,
            # given the ammount of pixels no nans along that row
            NoNan_acrX_cts = np.count_nonzero(~np.isnan(sev), axis=1)
            NoNan_acrX_cts = NoNan_acrX_cts.astype(int)

            np.seterr(invalid="ignore")

            # Express as percentage from pixels
            # ---------------------
            # OPTION1
            sev_pct = np.divide(sev_acrX, NoNan_acrX_cts) * 100
            sev_pct = np.round(sev_pct, 1)
            # ----------
            # OPTION2
            # sev_pct = np.divide(sev_acrX, width) * 100
            # sev_pct = np.round(sev_pct, 1  )

            med_acrX = np.nanmedian(arr_abs_grad, axis=1)

            # ---------Copy to result
            arr_sev_y[idx, :] = copy.deepcopy(sev_pct)
            arr_med_sev_acrX[idx, :] = copy.deepcopy(med_acrX)
            arr_threshold[idx] = copy.deepcopy(threshold)
            arr_NoNans[idx, :] = copy.deepcopy(NoNan_acrX_cts)

            if inps["plot_ind"] == True:  # or (debug==True):
                fn_out = os.path.join(
                    inps["out_dir_fig"], "abs_grad_az_{}.png".format(pair)
                )

                plot_ind(
                    arr_unw=arr_unw,
                    arr_abs_grad=arr_abs_grad,
                    sev_pct=sev_pct,
                    min_pct=80,  # min_pct,
                    fn_out=fn_out,
                    date12=pair,
                    orbit=inps["orbit"],
                )

            del sev_pct, med_acrX, arr_unw, arr_abs_grad, arr_coh, sev_acrX
            pbar.update(1)

    # Update
    inps["date12List_Skip"] = date12List_Skip

    # ----------------Begining Save Outputs ------------------------------#
    inps["stats_coh"] = stats_coh
    inps["stats_abs_grad"] = stats_abs_grad
    report_stats(inps)

    # All files saved are of dims=(pairs,Y)

    # Az gradient
    da_med_abs_grad = xr.DataArray(
        arr_med_sev_acrX,
        dims=("pair", "Y"),
        coords={
            "pair": date12List,
            "Y": np.arange(0, length, 1),
        },
    )
    da_med_abs_grad = da_med_abs_grad.where(da_med_abs_grad != 0)
    da_med_abs_grad = da_med_abs_grad.rename("abs_grad_az_mm")

    da_med_abs_grad.to_netcdf(
        fn_med_abs_grad,
        encoding={
            "abs_grad_az_mm": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 7,
            }
        },
    )

    # Severity
    da_sev_pct = xr.DataArray(
        arr_sev_y,
        dims=("pair", "Y"),
        coords={
            "pair": date12List,
            "Y": np.arange(0, length, 1),
        },
    )
    da_sev_pct = da_sev_pct.rename("sev_pct")
    da_sev_pct = da_sev_pct.where(da_sev_pct != 0)
    da_sev_pct.to_netcdf(
        fn_sev_pct,
        encoding={
            "sev_pct": {
                "dtype": "int16",
                "zlib": True,
                "complevel": 7,
                "_FillValue": -999,
            }
        },
    )

    # #Threshold per pair to define severity
    # da_treshold=xr.DataArray(
    #     arr_threshold,dims=('pair'),
    #     coords={'pair':date12List}
    #     )

    # da_treshold=da_treshold.rename("treshold_med_mm")
    # da_treshold.to_netcdf(fn_treshold,
    #                       encoding={
    #                           "treshold_med_mm": {
    #                               "dtype": "float32",
    #                               "zlib": True,
    #                               "complevel": 7,}
    #                           })

    # #Number of pixels with coh>coh_min
    da_CohPx_cts = xr.DataArray(
        arr_NoNans,
        dims=("pair", "Y"),
        coords={
            "pair": inps["date12List"],
            "Y": np.arange(0, arr_NoNans.shape[1], 1),
        },
    )
    da_CohPx_cts = da_CohPx_cts.rename("NoNanCounts")
    da_CohPx_cts.to_netcdf(
        fn_pxCoh_cts,
        encoding={"NoNanCounts": {"dtype": "int16", "zlib": True, "complevel": 7}},
    )

    # -----------------------
    # Analyze phase jump
    analyze_phase_jump(inps, da_sev_pct, da_CohPx_cts, da_med_abs_grad)

    # ---------------End Main Operations ---------------------------------#


# ------------------------------------#
def initiate_check(inps):
    # ----
    # Initiate parameters
    # ----

    inps["in_dir"] = os.path.abspath(inps["in_dir"])
    inps["fn_stack"] = os.path.join(inps["in_dir"], "inputs/ifgramStack.h5")
    inps["out_dir"] = os.path.join(inps["in_dir"], "pj_evaluation")
    inps["fn_coh_cts"] = os.path.join(inps["out_dir"], "maskCoh_cts.nc")
    inps["out_dir_fig"] = os.path.abspath(os.path.join(inps["out_dir"], "figs"))

    # ---
    # Define output name
    # ---
    # 3D (time,azimuth,range)
    # subfix_grad='abs_az_grad_mm.nc'
    # subfix_sev='sev.nc'

    # 2D (time, azimuth)
    subfix_sev_pct = "sev_pct.nc"
    subfix_nna = "coh_cts.nc"
    subfix_med = "med_az_grad_mm.nc"
    # subfix_tre='treshold_mm.nc'

    if inps["pair"] == None:
        # inps['fn_abs_grad_2D'] = os.path.join(inps["out_dir"], subfix_grad )
        # inps['fn_sev']= os.path.join(inps["out_dir"], subfix_sev)
        # ---
        inps["fn_pxCoh_cts"] = os.path.join(inps["out_dir"], subfix_nna)
        inps["fn_sev_pct"] = os.path.join(inps["out_dir"], subfix_sev_pct)
        inps["fn_med_abs_grad"] = os.path.join(inps["out_dir"], subfix_med)
        # inps['fn_tre']=os.path.join(inps["out_dir"],subfix_tre)
    else:
        # inps['fn_abs_grad_2D'] = os.path.join(inps["out_dir"],inps['pair'] + "_"+subfix_grad)
        # inps['fn_sev'] = os.path.join(inps["out_dir"], inps['pair']+ "_"+subfix_sev)
        # ---
        inps["fn_pxCoh_cts"] = os.path.join(
            inps["out_dir"], inps["pair"] + "_" + subfix_nna
        )
        inps["fn_sev_pct"] = os.path.join(
            inps["out_dir"], inps["pair"] + "_" + subfix_sev_pct
        )
        inps["fn_med_abs_grad"] = os.path.join(
            inps["out_dir"], inps["pair"] + "_" + subfix_med
        )
        # inps['fn_tre']=os.path.join(inps["out_dir"],inps['pair']+ "_"+subfix_tre)

    # ---
    # Check parameters
    # ---
    logging.info("Checking input parameters")

    if os.path.exists(inps["out_dir"]) is False:
        os.makedirs(inps["out_dir"])
        os.makedirs(inps["out_dir_fig"])
    # -
    if "plot_ind" not in inps.keys():
        inps["plot_ind"] = False

    skip = False
    if os.path.exists(inps["fn_stack"]) is False:
        logging.error("inputs/ifgramStack.h5 not found in parent directory.")
        skip = True
        return skip, inps

    elif os.path.exists(inps["fn_stack"]):
        atr = readfile.read_attribute(inps["fn_stack"])
        inps["orbit"] = atr["ORBIT_DIRECTION"]
        inps["length"] = int(atr["LENGTH"])
        inps["width"] = int(atr["WIDTH"])
        inps["wavelength"] = float(atr["WAVELENGTH"])
        if inps["subX"] != None:
            inps["subX"].sort()
            x0, x1 = inps["subX"][0], inps["subX"][1]
            # Check that x0 could be used:
            if (x0 < 0) or (x1 < 0):
                logging.error("No valid subset coordinates.")
                skip = True
                return skip, inps
            elif x1 <= inps["width"] - 1:
                # Define the width
                inps["width"] = x1 - x0
                logging.info("Calculations are done over AOI of %s pixels." % inps["width"])
            else:
                logging.error("No valid subset coordinates.")
                skip = True
                return skip, inps

        if "Y_FIRST" in atr.keys():
            logging.error("The stack must be in radar coordinates.")
            skip = True
            return skip, inps

    if inps["n_burst"] < 2:
        logging.error("Number burst < 2 - there are no burst overlapping areas.")
        skip = True
        return skip, inps
    # -
    files = glob.glob(os.path.join(inps["out_dir"], "*.nc"))
    if len(files) > 0:
        logging.warning("Output directory not empty. Results will be overwritten.")

        # skip=True
        return skip, inps
    return skip, inps


# --------------------------------#
def run(inps):
    skip, inps = initiate_check(inps)
    if skip == True:
        logging.info("Skip processing")
    else:
        if inps["pair"] == None:
            # Select input dates
            date12List = ifgramStack(inps["fn_stack"]).get_date12_list(dropIfgram=False)
            logging.info("Total of interferograms found: {} ".format(len(date12List)))

            inps["date12List"] = date12List

            ref = [date12.split("_")[0] for date12 in date12List]
            sec = [date12.split("_")[1] for date12 in date12List]
            dates = np.unique((ref + sec))
            dates = list(dates)
            inps["dates"] = dates.sort()
        else:
            date12List = [inps["pair"]]
            inps["date12List"] = date12List

        inps = unwrap_phase2azimuth_gradient(inps)


run(inps)
end = time.time()
logging.info("Processing Time %s minutes" % (np.round((end - start) / 60, 2)))
