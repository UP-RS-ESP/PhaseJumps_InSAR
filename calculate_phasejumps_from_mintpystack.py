#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------#
# Author Sofia Viotto (viotto1@uni-potsdam.de), Bodo Bookhagen
# V0.1 Oct-2024
# V0.2 Jan-2025

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os, glob
from mintpy.objects import ifgramStack
from mintpy.utils import readfile
# import seaborn as sns
import argparse
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt
import numba as nb
import datetime
#
import copy,logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

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
synopsis = "Extract azimuth gradient from unwrapped phase in radar coordinates and find phase jumps"
EXAMPLE = """
    python calculate_phasejumps_from_mintpystack.py --in_dir /path/mintpy  --plot-ind --n-burst 9 \n 
    python calculate_phasejumps_from_mintpystack.py --in_dir /path/mintpy --pair 20160524_20160711 --n-burst 9  --min-pct 85 \n 
    
    *****************************************************************
    
    The number of bursts is listed during ISCE processing. If you did not save the logs, you 
    can easily identify the number of bursts by looking into the reference directory of your ISCE output. For example:
    ls -1 reference/IW1/*.vrt | wc -l
    will give you the number of bursts if you have proceesed sub-swath IW1. This is similar to looking into the secondary directories:
    ls -1 secondarys/20210412/IW1/burst_0*.vrt | wc -l    

    Note: \n 
    1)The program requires full extension in the azimuthal direction (i.e. the stacks must not be subset along the azimuthal dimension). \n
    2)This program is designed to detect phase jumps within a single sub-swath (IW1, IW2 or IW3). 
It is not compatible yet with stacks created by merging sub-swaths (e.g., IW2-IW3 interferograms).
    
    *****************************************************************
    
    References
    1) Wang et al., 2017 "Improving burst aligment in tops interferometry with BESD",
    10.1109/LGRS.2017.2767575
    2) Zhong et al., 2014 "A Quality-Guided and Local Minimum Discontinuity Based 
    Phase Unwrapping Algorithm for InSAR InSAS Interferograms", 10.1109/LGRS.2013.2252880
    ******************************************************************
    
"""

parser = argparse.ArgumentParser(description=synopsis, epilog=synopsis, usage=EXAMPLE)
parser.add_argument(
    "--inDir",
    "-i",
    dest="in_dir",
    help="Input folder containning input/ifgramStack.h5 in RADAR COORDINATES",
    default=os.getcwd(),
)
parser.add_argument(
    "--plot-ind",
    dest="plot_ind",
    help="plot individual interferograms next to gradient along the azimuth direction",
    action="store_true",
)
parser.add_argument("--pair", "-p", default=None, help="Perfom calculations only for pair from the file inputs/ifgramStack.h5", dest="pair")
parser.add_argument(
    "--n-burst", "-n",
    dest="n_burst",
    type=int,
    help="Specify the number of bursts expected within the dataset.",
)
parser.add_argument(
    "--min-pct",
    default=85.0,
    help="Minimun percentage of pixels jumping along a row to be defined as phase jump. It is not recomendable to change the thershold, as the rows may be note reliable",
    dest="min_pct",
    type=float
)
parser.add_argument("--sub-x",default=None, help=' Define the area of interest along x',
                    dest="subX",
                    nargs=2,
                    type=int
                    )
args = parser.parse_args()
inps = args.__dict__


# ------------------------------------#
def initiate_check(inps):
    # ----
    # Initiate parameters
    # ----
      
    inps["in_dir"] = os.path.abspath(inps["in_dir"])
    inps["fn_stack"] = os.path.join(inps["in_dir"], "inputs/ifgramStack.h5")
    inps["out_dir"] = os.path.join(inps["in_dir"], "jumps_eval")
    inps["out_dir_fig"] = os.path.abspath(os.path.join(inps["out_dir"], "fig"))
    #print(inps)

    #---
    #Define output name
    #---
    #3D (time,azimuth,range)
    # subfix_grad='abs_az_grad_mm.nc'
    # subfix_sev='sev.nc'

    #2D (time, azimuth)    
    subfix_sev_pct="sev_pct.nc"
    subfix_nna="nonan_cts.nc"
    subfix_med="med_az_grad_mm.nc"
    subfix_tre='treshold_mm.nc'
    
    if inps["pair"] == None:
        # inps['fn_abs_grad_2D'] = os.path.join(inps["out_dir"], subfix_grad )
        # inps['fn_sev']= os.path.join(inps["out_dir"], subfix_sev)
        #---
        inps['fn_nna_cts'] = os.path.join(inps["out_dir"], subfix_nna)
        inps['fn_sev_pct'] = os.path.join(inps["out_dir"], subfix_sev_pct)
        inps['fn_med_abs_grad']=os.path.join(inps["out_dir"], subfix_med)
        inps['fn_tre']=os.path.join(inps["out_dir"],subfix_tre)
    else:
        # inps['fn_abs_grad_2D'] = os.path.join(inps["out_dir"],inps['pair'] + "_"+subfix_grad)
        # inps['fn_sev'] = os.path.join(inps["out_dir"], inps['pair']+ "_"+subfix_sev)
        #---
        inps['fn_nna_cts'] = os.path.join(inps["out_dir"], inps['pair']+ '_'+subfix_nna)
        inps['fn_sev_pct'] = os.path.join(inps["out_dir"],inps['pair'] + "_"+subfix_sev_pct)
        inps['fn_med_abs_grad']=os.path.join(inps["out_dir"], inps['pair']+ "_"+subfix_med)
        inps['fn_tre']=os.path.join(inps["out_dir"],inps['pair']+ "_"+subfix_tre)
    
    #---
    #Check parameters
    #---
    logging.info('Checking input parameters')
    if os.path.exists(inps["out_dir"]) is False:
            os.makedirs(inps["out_dir"])
            os.makedirs(inps["out_dir_fig"])
    #-
    if "plot_ind" not in inps.keys():
        inps["plot_ind"] = False
        
    
            
    skip = False
    if os.path.exists(inps["fn_stack"]) is False:
        logging.error("inputs/ifgramStack.h5 not found in parent directory.")
        skip = True
        return skip,inps
   
        
    elif os.path.exists(inps["fn_stack"]):
        atr = readfile.read_attribute(inps["fn_stack"])
        inps["orbit"] = atr["ORBIT_DIRECTION"]
        inps['length']=int(atr['LENGTH'])
        inps['width']=int(atr['WIDTH'])
        inps['wavelength']=float(atr["WAVELENGTH"])
        if inps['subX'] != None:
            inps['subX'].sort()
            x_0,x_1 = inps['subX'][0],inps['subX'][1]
            #Check that x_0 could be used:
            if (x_0) <0 or (x_1<0):
                logging.error("No valid subset coordinates")
                skip=True
                return skip, inps
            elif x_1 <= inps['width'] -1:
                #Define the width
                inps["width"] = x_1-x_0
                logging.info('Calculations over AOI of %s pixels'%inps['width'])
            else:
                logging.error("No valid subset coordinates")
                skip=True
                return skip, inps   
            
        if "Y_FIRST" in atr.keys():
            logging.error(
                "The stack must be in radar coordinates"
            )
            skip = True
            return skip,inps
            
    if inps['n_burst'] < 2:
        logging.error("Number burst < 2, then there are not burst overlapping areas")
        skip=True
        return skip,inps
    #-  
    files = glob.glob(os.path.join(inps["out_dir"], "*.nc"))
    if len(files) > 0:
        logging.warning("Output directory not empty. Results will be overwritten") 
        #skip=True
        return skip,inps
    return skip, inps   


# -------Plot fuctions
def PlotData_UnwGradient(arr_unw,arr_abs_grad,sev_pct,min_pct,fn_out,date12,orbit):
    fig_size = (10, 7)

    title = "Pair %s (Mask Coherence)" % date12

    # Plot
    fig, axs = plt.subplots(ncols=3,
        figsize=fig_size,
        sharex=False,
        sharey=True,
        gridspec_kw={'width_ratios': [3, 3, 1]}
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
        aspect='auto'
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
    vmax = np.nanpercentile(arr_abs_grad, 98)
    vmin = np.nanpercentile(arr_abs_grad, 2)
    gradPlot = axs[1].imshow(
        arr_abs_grad, cmap="viridis", interpolation="none", vmin=vmin, vmax=vmax,
        aspect='auto'
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


#----------------Aux fuctions
def group_coord(y, fr, ovlp_reg):
    
    y_gps = np.split(y, np.where(np.diff(y) > ovlp_reg/2)[0] + 1)
    fr_gps = np.split(fr, np.where(np.diff(y) > ovlp_reg/2)[0] + 1)
    del y,fr
    
    y_final = []
    
    for gp, fr_gp in zip(y_gps, fr_gps):
        if gp.shape[0] > 1:
            try:
                # #Find the coordinate that have the max frequency
                y_final.append(copy.deepcopy(gp[fr_gp == np.max(fr_gp)].item()))
                
            except:
                #If the frequency is the same append the first element of the list
                y_final.extend(copy.deepcopy(gp))
        else:
            y_final.append(gp[0])
    return y_final
   

def get_phase_jump_cord(da_sev_pct,min_pct,ovlp_reg,n_burst):
    da_sev_pct2=copy.deepcopy(da_sev_pct)
    da_sev_pct2 = da_sev_pct2.where(da_sev_pct2 > min_pct)

    # Drop pairs without rows suspected of phase jump
    da_sev_pct2 = da_sev_pct2.dropna(dim="pair", how="all")
    
    # Drop coordinates without da_sev_pct2
    da_sev_pct2 = da_sev_pct2.dropna(dim="Y", how="all")
    
    #Keep the list of filtered dates
    date12_filt = list(da_sev_pct2.pair.values)


    # # -----Rough position guided by regular coordinates
    # logging.info(
    #     "Estimating position of Burst Overlap based on sample of size %s unwrap phase"
    #     % da_sev_pct2.pair.shape[0]
    # )
    
    #Container of rough coordinates along azimuth
    y_cord_rough = []

    for idx, date12 in enumerate(date12_filt):
        
        #Copy pair to avoid modify the original
        sev_pct_pair = da_sev_pct2.isel(pair=idx).copy()
        #Keep coordinates
        y_cord_sev_pair = sev_pct_pair.dropna(dim="Y").Y.values
        #----------------------------------------------------------#
        #Calculate shift from coordinates to regular coordinates
        # dy = np.abs( y_cord_sev_pair- (y_cord_sev_pair // ovlp_reg) * ovlp_reg)

        # # Given filter + multilooking, 
        # # allow a max shift of certain pixels between the regular position
        # # and the position where the phase jump is found
 
        # y_cord_sev_pair = y_cord_sev_pair[  dy < ovlp_reg/3 ]
        #--------------------------------------------------------#
        # # Remove coordinates at the upper border
        y_cord_sev_pair = y_cord_sev_pair[y_cord_sev_pair < ((n_burst * ovlp_reg) - ovlp_reg/3)]
        # Remove coordinates at the lower border
        y_cord_sev_pair=y_cord_sev_pair[y_cord_sev_pair>ovlp_reg/3]
        y_cord_rough.extend(copy.deepcopy(list(y_cord_sev_pair)))
        
        #Clean
        del y_cord_sev_pair, sev_pct_pair
        

    # ----------------
    # STAGE 3:
    # --Refinement of positions of phase jumps based on frequency across of pairs
    # ----------------
    #logging.info("Refining coordinates based on frequency of phase jumps")
    y, fr = np.unique(y_cord_rough, return_counts=True)

    # Group closer coordinates
    y_final=group_coord(y, fr, ovlp_reg)
    
    
    return y_final

def analyze_phase_jump(inps,da_sev_pct,da_NoNan,da_med_abs_grad):
    """
    Analyzes and identifies phase phase jumps in overlapping areas of bursts.
    Phase jumps in the overlapping area are distinguished by the fact that
    they are located at approximately regular intervals.

    The programm reads sev files and identifies phase phase jumps at regular intervals.
    Coordinates found are reported.

    Parameters:
    -----------
    inps : dict
        A dictionary containing the following keys:
        - 'date12List' : list
            List of date pairs. Format 'YYYYMMDD_yyyymmdd'

        - 'min_pct' : float
            Minimum height percentage to consider a peak of detected pixels as a phase jump, default is 85% of the pixels along the row (range direction).

        - 'in_dir' : str
            Directory path where output reports will be saved.
        - 'in_dir_arr_1d' : str
            Directory path containing 1D sev files.
        - 'in_dir_arr_2d' : str
            Directory path containing 2D sev files.
        - 'out_dir_fig' : str
            Directory path where output figures will be saved.
        - 'fn_absolute_azimuth_gradient' : str
            Filename of the 2D azimuth gradient file.
        - 'n_burst' : int
            Number of bursts in the dataset.

    """

    # Input
    date12List = inps["date12List"]
    n_burst = inps["n_burst"]
    min_pct = inps["min_pct"]
    #
    length=inps['length']

    # Output reports
    out_fn_report = os.path.join(inps["in_dir"], "phase_jumps_per_date.txt")
    out_fn_summary = os.path.join(inps["in_dir"], "summary_phase_jumps_y_cordinates.txt")
    out_fn_excList_ifgs = os.path.join(inps["in_dir"], "exclude_listdate12_interferograms_by_phase_jump.txt")
    out_fn_excList_dat = os.path.join(inps["in_dir"], "exclude_dates_by_phase_jumps.txt")
    out_fn_stats = os.path.join(inps["out_dir"], "stats_absolute_gradient.txt")
    
    # ----------------
    # STAGE 1:
    # Determine regular distance of overlapping areas
    # ---------------
    ovlp_reg = length // n_burst
    

    # ------------------
    # STAGE 2:
    # Store coordinates with phase jumps in each sev file
    # ------------------
    # Pairs with representative rows are used to define the coordinates of phase jumps.
    # Criteria:
        # 1) The number of NoNan pixels is at least greater than the 10th percentile 
        #    of the total pixels available from all rows and all pairs.
        # This is better than using 10% of the length (number of rows) as a threshold, as it accounts for 
        # coherence across all pairs.
        # 2) The percentage of pixels at certain row with an absolute gradient > median exceeds 85%.
        # 3) If a pair does not meet both (1) and (2), it is discarded.
        # 4) Only coordinates that satisfy (1) and (2) will be analyzed.
    #---------------------------#
    da_NoNan=da_NoNan.where(da_NoNan!=-999)
    NoNan_10p=np.nanpercentile(da_NoNan.data, 10)
    if debug==True:
        print('Reliable rows have ',NoNan_10p)
  
    da_sev_pct=da_sev_pct.where(da_NoNan>NoNan_10p)
    
    y_iter=[]
    with tqdm(total=10, desc="Iteration",ncols=100, bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for n in range(0,10):
            min_pct_1=min_pct+n*(100-min_pct)/10
            y_iter.extend(get_phase_jump_cord(da_sev_pct,min_pct_1,ovlp_reg,n_burst))
            pbar.update(1)
            
    y,fr=np.unique(y_iter, return_counts=True)
    y_final=group_coord(y, fr, ovlp_reg)
       
    
    logging.info("Coordinates of phase jumps %s " % y_final)
    
    #--------------#    
    da_sev_pct = da_sev_pct.sel(Y=np.asarray(y_final))
    da_sev_pct = da_sev_pct.dropna(dim="pair", how="all")
    
    da_med_abs_grad=da_med_abs_grad.sel(Y=np.asarray(y_final))
    
    # ----------------------------------------#
    if debug==True:
        #Add the median phase jump per coordinate
        values = []
        for pair in date12List:
            arr = da_med_abs_grad.sel(pair=pair).copy()
            arr = arr.sel(Y=np.asarray(y_final))
            arr = arr.data.flatten()
            arr = np.round(arr, 2)
            arr = list(arr)

            arr = [str(i) for i in arr]

            values.append("," + ",".join(arr) + "\n")
            del arr,pair
            
    
        # Open stats and add the new stats:
    
        logging.info("Median Absolute Gradient per Coordinate saved at %s " % out_fn_stats)
    
        with open(out_fn_stats, "r") as fl:
            lines = fl.readlines()
            fl.close()
        
        y_txt = ["AbsGrad_Median_Y-" + str(int(i)) + "[mm]" for i in y_final]
        lines[10] = lines[10].replace("\n", "," + ",".join(y_txt) + "\n")
    
        # Add the statis
        subset_lines = copy.deepcopy(lines[11:])
        subset_lines = [i.replace("\n", j) for i, j in zip(subset_lines, values)]
        lines[11:] = subset_lines
        with open(out_fn_stats, "w") as fl:
            for line in lines:
                fl.write(line)
            fl.close()
        del subset_lines,y_txt,lines,values

    # --------------------#
    # STAGE 4
    # -------------------#
    #Find the pairs with median ds abs grad larger than the 95th percentile
    #logging.info("Looking at the distrbution of the median absolute gradient per row %s " % out_fn_stats)
    med_threshold = 1.0#np.nanpercentile(da_med_abs_grad.data, 99)
    
    if debug==True:
        plt.figure()
        x=da_med_abs_grad.data.flatten()
        x=x[x!=np.nan]
        plt.hist(x)
        plt.axvline(med_threshold,c='red')
        plt.axvline(x=1.0,c='red')
        plt.savefig(os.path.join(inps['out_dir'],'histogram_med_absolute_gradient.png'),dpi=100)

    da_med_abs_grad_filt = da_med_abs_grad.where(
        da_med_abs_grad >= med_threshold
    )
    da_med_abs_grad_filt = da_med_abs_grad_filt.dropna(dim="pair", how="all")
    
    #Keep the dates
    date12_filt_final = list(da_med_abs_grad_filt.pair.values)

    logging.info("Phase jump is significant if the median jump of the row is >%s mm\n"
    % np.round(med_threshold, 2)+
        " Total of pairs found with significant phase jumps: %s"
        % len(date12_filt_final)
    )

    # ----------------------------------------------#
    # Report phase jumps
    # ----------------------------------------------#
    #--------------Beging Report
    y_summary = []
    
    report_txt = []
    y_pj_pair = []
    logging.info(
        "Reporting phase jumps %s"
        % out_fn_report
    )
    for idx, date12 in enumerate(date12_filt_final):

        da_med_abs_grad_pair = da_med_abs_grad_filt.sel(pair=date12).copy()
        y_pj_pair = list(da_med_abs_grad_pair.dropna(dim="Y").Y.values)

        if len(y_pj_pair) > 0:
            # Store
            y_summary.extend(copy.deepcopy(y_pj_pair))
            
            #Transform to string for reporting
            y_pj_pair = [str(i) for i in y_pj_pair]
            spacing = "\t" * ((n_burst // 2 - len(y_pj_pair) // 2))
            
            # Calculate the average phase jump
            avg_pj = np.nanmean(da_med_abs_grad_pair.data)
            
            report_txt.extend(
                [
                    date12
                    + "\t"
                    + str(len(y_pj_pair))
                    + "\t\t"
                    + ",".join(y_pj_pair)
                    + spacing
                    + str(np.round(avg_pj, 2))
                ]
            )
        else:
            # Remove date12 if there is not date to report
            date12_filt_final = [
                i for i in date12_filt_final if i != date12
            ]
            continue
        del da_med_abs_grad_pair

    
    if len(report_txt) > 0:

        header = ("# Pairs found with systematic phase jumps \n"
        "# N_Disc: Number of phase jumps per pair \n"
        "# Az_cord: Azimuth coordinate with phase jump\n"
        "# Avg_Ph_Jump_mm: Average phase jump \n" )

        # -Report each pair, number of phase jumps and coordinates
        with open(out_fn_report, "w") as fl:
            fl.write(header)
            fl.write(
                "#\t DATE12 \tN_Disc\t\tAz_cord\t\t\tAvg_Ph_Jump_mm\n"
            )
            for item in report_txt:
                fl.write("%s\n" % item)
            fl.close()

    # Provide a summary of the coordinates found
    y_, cts_ = np.unique(y_summary, return_counts=True)
    with open(out_fn_summary, "w") as fl:
        fl.write("#Az_cord\tCounts\n")
        for y_i, cnt_i in zip(y_, cts_):
            fl.write("%s\t%d\n" % (y_i, cnt_i))
            
    del y_, cts_
    # #------------------
    # #Create a list of interferograms

    with open(out_fn_excList_ifgs, "w") as fl:
        fl.write(",".join(date12_filt_final))

    # #--------------------------
    # Create a list to exclude dates from further processing based on how many interferograms
    # With phase jumps were found per date
    dates = [pair.split("_")[0] for pair in date12List]
    dates.extend([pair.split("_")[1] for pair in date12List])
    dates, fr_interferograms = np.unique(dates, return_counts=True)

    dateswithphase_jumps = [i.split("_")[0] for i in date12_filt_final]
    dateswithphase_jumps.extend(i.split("_")[1] for i in date12_filt_final)
    dateswithphase_jumps, fr_ifgs_phase_jumps = np.unique(
        dateswithphase_jumps, return_counts=True
    )

    proportion = np.asarray(
        [
            n_ifgs_phase_jump * 100 / fr_interferograms[dates == date_i]
            for date_i, n_ifgs_phase_jump in zip(
                dateswithphase_jumps, fr_ifgs_phase_jumps
            )
        ]
    ).flatten()

    dd = list(dateswithphase_jumps[proportion > 50])
    # logging.info(dd)
    with open(out_fn_excList_dat, "w") as fl:
        fl.write(",".join(dd))


def save_global_stats2txt(inps):

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
    "# No Data Values (Zero Values) were excluded from all calculations.\n"
    "# Coherence statistics were derived from all non-NaN pixels.\n"
    "# Pixels with coherence < 0.75 were masked out during the calculation of absolute azimuth gradient and corresponding severity.\n"
    "# The number of masked-out pixels varies between pairs.\n"
    "## Column Names/Prefixes:\n"
    "# Btemp: Temporal Baseline\n"
    "# Coh: Coherence\n"
    "# AbsGrad: Absolute gradient along the azimuth direction and across range \n"
    "#Med: Median, Std: Standard deviation\n\n"
 )

    stats_name = ["AbsGrad_Median_mm", "AbsGrad_Mean_mm", "AbsGrad_Std_mm"]
    # Prepare name of columns
    colCoh = copy.deepcopy(stats_name)
    colCoh = [i.replace("AbsGrad", "Coh").replace("_mm", "") for i in colCoh]
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


@nb.njit()
def calculate_stats_arrays(ds_unw,ds_coh):

    stats_abs_grad = np.empty((3), dtype=np.float32)
    stats_coh = np.empty((3), dtype=np.float32)

    stats_abs_grad[0] = np.round(np.nanmedian(ds_unw), 2)
    stats_abs_grad[1] = np.round(np.nanmean(ds_unw), 2)
    stats_abs_grad[2] = np.round(np.nanstd(ds_unw), 2)
    
    stats_coh[0] = np.round(np.nanmedian(ds_coh), 2)
    stats_coh[1] = np.round(np.nanmean(ds_coh), 2)
    stats_coh[2] = np.round(np.nanstd(ds_coh), 2)    
    
        
    return stats_abs_grad, stats_coh


def readData2VerticalGradient(inps):
    # ----------------
    #Input
    #----------------
    fn_stack = inps["fn_stack"]
    #Y=length=azimuth coordinate
    #X=width=range coordinates
    length=inps['length']
    width=inps['width']
    date12List=inps['date12List']
    wavelength=inps['wavelength']
    phase2range = wavelength / (4.0 * np.pi)
    min_pct=inps['min_pct']
    # ----------------
    #Output
    #-----------------
    #3D (time, azimuth, range)
    # fn_abs_grad=inps['fn_abs_grad_2D']
    # fn_sev = inps['fn_sev']
    
    #2D (time, azimuth)
    fn_nna_cts = inps['fn_nna_cts']
    fn_sev_pct = inps['fn_sev_pct']
    fn_med_abs_grad=inps['fn_med_abs_grad']
    fn_treshold=inps['fn_tre']
        

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


    #---------------------------------#
    #Bodo: load the whole dataset is not a good idea for heavy files 
    # The memory may be not enough. It's better and faster
    # to apply the fuction one pair at the time. 
    #-------------------------------#
    
    
    #Create containers for arrays
    arr_med_sev_acrX=np.zeros((len(date12List),length),dtype=float)
    arr_sev_y=np.zeros((len(date12List),length),dtype=float)
    arr_threshold=np.zeros(len(date12List),dtype=float)
    arr_NoNans=np.zeros((len(date12List),length),dtype=int)
    arr_NoNans-=999
    #Create containers for stats
    stats_abs_grad=np.zeros((len(date12List),3),dtype=float)
    stats_coh=np.zeros((len(date12List),3),dtype=float)
    logging.info('Calculating Absolute Gradient in the Azimuth Direction')
    with tqdm(total=len(date12List), desc="Pairs processed",ncols=100, bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        #Loop to generate sev    
        for idx, pair in enumerate(date12List):
            
            #---------------------------------#
            # Note: Loading the entire dataset is not a good idea for large files.
            # The memory might not be sufficient. It is better and faster
            # to apply the function one pair at a time.
            #---------------------------------#
            #Read data
            if inps['subX']!=None:
                #(x0, y0, x1, y1)
                x0,x1=np.min(inps['subX']),np.max(inps['subX'])
                bbox=[x0,0,x1,length]
            
            else:
                bbox=None
            
            arr_unw = readfile.read(fn_stack, 
                               datasetName='unwrapPhase-'+pair,
                               box=bbox)[0]
            arr_coh= readfile.read(fn_stack, 
                              datasetName='coherence-'+pair,
                              box=bbox)[0]
            #Apply mask
            arr_unw[arr_coh<0.75]=np.nan
        
            #Remove no data
            arr_coh[arr_coh==0]=np.nan
            arr_unw[arr_unw==0]=np.nan
        
            #Calculate absolute gradient
            #Do not prepend a zero, as the first line is already zero
            arr_abs_grad=np.zeros(arr_unw.shape, dtype=float)
            arr_abs_grad[1:,:]=np.abs(np.diff(arr_unw,axis=0))
            arr_abs_grad[:2,:]=np.nan
            #  Set the last lines to zero, to avoid border effects
            arr_abs_grad[-2:,:]=np.nan
            
        
            #Convert to displacement and express in milimeters
            arr_abs_grad *= phase2range
            arr_abs_grad *= 1000
            #Set outliers to nan as well, BEFORE deriving statistics
            p99=np.nanpercentile(arr_abs_grad,99)
            if debug==True:
                logging.info('Removing Outliers  beyond value %s mm \n' %p99)
                logging.info('Removing Outliers  with max value %s mm \n' %np.nanmax(arr_abs_grad))
            #-
            arr_abs_grad[arr_abs_grad>p99]=np.nan
        
        
            #Obtain stats 
            stats_abs_grad[idx,:],stats_coh[idx,:]=calculate_stats_arrays(arr_abs_grad,arr_coh)
            
        
        
            # -----------------------------------------
            # Calculate the sev (magnitude) of the phase jump
            # -----------------------------------------
            # Calculate the sev , as the division
            # between the Absolute Azimuth Gradient and  the threshold,
            # then round the values to zero decimals.
            #
            #  sev(time_i)=round(absolute_gradient(time_i)/threshold(time_i))
            #with threshold= median(absolute_gradient(time_i))
            #
            # By rounding, 0 represent gradients below and far away from ratio gradient to threshold
            # 1=phase jumps close or above 1 from the threshold
            #-------------------------------------------------
            #First threshold it is if the difference is larger than the median gradient of the track
            threshold=stats_abs_grad[idx,0]
        
            #Dims (row-1,col)=(azimuth,range)=(y,x)
            sev=np.zeros(arr_abs_grad.shape,dtype=float)
            sev=np.where(arr_abs_grad>=threshold,1,sev)
            mask=np.isnan(arr_abs_grad)
            sev[mask]=np.nan
            #sev = np.round(np.divide(arr_abs_grad, threshold), 0)
            if debug==True:
                print(np.nanmin(sev),np.nanmean(sev),np.nanmax(sev))
                plt.figure()
                plt.hist(sev[sev!=np.nan],bins=5)
                out=os.path.join(inps['out_dir_fig'],'hist_sev_%s.png'%pair)
                plt.savefig(out,dpi=100)
                plt.close()
            sev[sev > 1] = 1
        
            #Dims (row-1)
            sev_acrX = np.nansum(sev, axis=1)
        
            #Count NoNans pixels to 1) compute percentage along the row, and 
            # 2) determine if the row is reliable,
            #given the ammount of pixels no nans along that row
            NoNan_acrX_cts = np.count_nonzero(~np.isnan(sev), axis=1)
            NoNan_acrX_cts = NoNan_acrX_cts.astype(int)
            
            np.seterr(invalid="ignore")
            #Express as percentage from pixels
            sev_pct = np.divide(sev_acrX, NoNan_acrX_cts) * 100
            sev_pct = np.round(sev_pct, 1)
                  
            med_acrX=np.nanmedian(arr_abs_grad,axis=1)
        
            #---------Copy to result
        
            arr_sev_y[idx,:]=copy.deepcopy(sev_pct)
            arr_med_sev_acrX[idx,:]=copy.deepcopy(med_acrX)
            arr_threshold[idx]=copy.deepcopy(threshold)
            arr_NoNans[idx,:]=copy.deepcopy(NoNan_acrX_cts)
        
            if (inps['plot_ind']==True) or (debug==True):
                fn_out=os.path.join(inps['out_dir_fig'],'abs_grad_az_{}.png'.format(pair))
            
                PlotData_UnwGradient(arr_unw=arr_unw, 
                                  arr_abs_grad=arr_abs_grad, 
                                  sev_pct=sev_pct, 
                                  min_pct=min_pct, 
                                  fn_out=fn_out, 
                                  date12=pair, 
                                  orbit=inps['orbit'])
            
            
            del sev_pct,med_acrX,arr_unw,arr_abs_grad,arr_coh,sev_acrX
            pbar.update(1)
            
    inps['stats_coh']=stats_coh
    inps['stats_abs_grad']=stats_abs_grad
    save_global_stats2txt(inps)      
  



    # #----------------Begining Save Outputs ------------------------------#
    # #--------------------------#
    # #Save
    # #--------------------------#
    logging.info("Saving relevant files")
    #---------
    #All files saved are of dims=(pairs,Y)
    #
    #Az gradient
    da_med_abs_grad = xr.DataArray(
                arr_med_sev_acrX,
                dims=("pair", "Y"),
                coords={
                    "pair": date12List,
                    "Y": np.arange(0, length, 1),
                    
                },
            )
    da_med_abs_grad=da_med_abs_grad.where(da_med_abs_grad!=0) 
    da_med_abs_grad=da_med_abs_grad.rename("abs_grad_az_mm")
    
    da_med_abs_grad.to_netcdf(fn_med_abs_grad,
                              encoding={
                                  "abs_grad_az_mm": {
                                      "dtype": "float32",
                                      "zlib": True,
                                      "complevel": 7,
                                  }
                              }
                         )
    
     
    
    #Severity
    da_sev_pct= xr.DataArray(
                arr_sev_y,
                dims=("pair", "Y"),
                coords={
                    "pair": date12List,
                    "Y": np.arange(0, length, 1),
                    
                },
            )
    da_sev_pct=da_sev_pct.rename("sev_pct")
    da_sev_pct=da_sev_pct.where(da_sev_pct!=0)
    da_sev_pct.to_netcdf(fn_sev_pct,
                     
                     encoding={
                    "sev_pct":{"dtype": "int16",
                    "zlib": True,
                    "complevel": 7,
                    "_FillValue": -999,
                                                            }
                                                })
    
    
    #Threshold per pair to define severity
    da_treshold=xr.DataArray(
        arr_threshold,dims=('pair'),
        coords={'pair':date12List}
        )

    da_treshold=da_treshold.rename("treshold_med_mm")
 
   
    da_treshold.to_netcdf(fn_treshold,
                          encoding={
                              "treshold_med_mm": {
                                  "dtype": "float32",
                                  "zlib": True,
                                  "complevel": 7,}
                              })

    
    # NonNans
    da_NoNan = xr.DataArray(
            arr_NoNans,
            dims=("pair", "Y"),
            coords={
                "pair": inps["date12List"],
                "Y": np.arange(0, arr_NoNans.shape[1], 1),
            },
        )
    da_NoNan = da_NoNan.rename("NoNanCounts")
    da_NoNan.to_netcdf(
            fn_nna_cts,
            encoding={"NoNanCounts": {"dtype": "int16", "zlib": True, "complevel": 7}},
        )
   
    #-----------------------
    #Analyze phase jup
    analyze_phase_jump(inps, da_sev_pct, da_NoNan, da_med_abs_grad)

    #---------------End Main Operations ---------------------------------#


    return inps



def run(inps):
    skip, inps = initiate_check(inps)
    if skip == True:
         logging.info("Skip processing")
    else: 
        if inps["pair"] == None:

            # Select input dates
            date12List = ifgramStack(inps["fn_stack"]).get_date12_list(
                        dropIfgram=False
                    )
            logging.info("Total of interferograms found: {} ".format(len(date12List)))
        
            inps["date12List"] = date12List
           
            ref = [date12.split("_")[0] for date12 in date12List]
            sec = [date12.split("_")[1] for date12 in date12List]
            dates = np.unique((ref + sec))
            dates=list(dates)
            inps["dates"] = dates.sort()
        else:
            date12List=[inps['pair']]
            inps['date12List']=date12List
        
        inps = readData2VerticalGradient(inps)

run(inps)
end = time.time()
logging.info('Processing Time %s minutes'%(np.round((end - start)/60,2)))

