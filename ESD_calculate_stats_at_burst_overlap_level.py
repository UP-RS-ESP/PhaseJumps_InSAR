#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:37:41 2024

@author: Sofia Viotto
"""

# #----------------------------------------

# #- Bibliography

# R. Scheiber and A. Moreira, "Coregistration of interferometric SAR images using spectral diversity," in IEEE Transactions on Geoscience and Remote Sensing, vol. 38, no. 5, pp. 2179-2191, Sept. 2000, doi: 10.1109/36.868876

# H. Fattahi, P. Agram and M. Simons, "A Network-Based Enhanced Spectral Diversity Approach for TOPS Time-Series Analysis," in IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 2, pp. 777-786, Feb. 2017, doi: 10.1109/TGRS.2016.2614925

# N. Yagüe-Martínez et al., "Interferometric Processing of Sentinel-1 TOPS Data," in IEEE Transactions on Geoscience and Remote Sensing, vol. 54, no. 4, pp. 2220-2234, April 2016, doi: 10.1109/TGRS.2015.2497902.

# #-------------------------------------------------
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os, glob, logging
import xarray as xr
import argparse
import matplotlib.pyplot as plt
from argparse import RawTextHelpFormatter

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------
EXAMPLE = """example:
python ESD_calculate_stats_at_burst_overlap_level.py \
    --inDir /Home/ISCE_processing/ESD 
    
python ESD_calculate_stats_at_burst_overlap_level.py \
    --inDir /Home/ISCE_processing/ESD  --subwath IW1 
python ESD_calculate_stats_at_burst_overlap_level.py \
    --inDir /Home/ISCE_processing/ESD  --subwath IW1 IW2 
"""

DESCRIPTION = """
Calculates statistics from Enhanced Spectral Diversity (ESD) files used to proper align the sences (stackSentinel.py tool from ISCE).  
The script processes ESD files to extract median values, standard deviations, and coherence points, at burst overlaps levels. 


Oct-2024, Sofia Viotto (viotto1@uni-potsdam.de), Bodo Bookhagen



"""

# -----------------------
# Threshold based on
# --
# Yagüe-Martinez et al, 2016
# "Coregistration Accuracy"
# --
# Fatttahi et al., 2017
# "A Network-Based Enhanced Spectral Diversity
# Approach for TOPS Time-Series Analysis"
# The a misregistration along the azimuth direction
# is about 0.01+/-0.001
# Therefore the median should not
# differ more than 0.001
# -------------------------
threshold = 0.0009

# ---------------------
parser = argparse.ArgumentParser(
    description=DESCRIPTION, epilog=EXAMPLE, formatter_class=RawTextHelpFormatter
)
parser.add_argument("--inDir", "-i", dest="inDir", help="Full path to the ESD folder")
parser.add_argument(
    "--subswath",
    "-s",
    dest="subswath",
    help="Optional. Define a sub-swath to calculate statistics",
    nargs="+",
    default=None,
)
args = parser.parse_args()

# ------------------------------------------------#


def check_input_directories(inps):
    skip = False
    # Check if the main input directory exists
    if os.path.exists(inps["inDir"]) is False:
        skip = True
        # logging.info('Input Directory does not exist \n')
        logging.info("Input Directory does not exist.")
        return inps, skip
    # Create ESD_azimuth_offsets directory if not found
    elif os.path.exists(os.path.join(inps["inDir"], "ESD_azimuth_offsets")) is False:
        os.mkdir(os.path.join(inps["inDir"], "ESD_azimuth_offsets"))

    # Check if the ESD directory exists
    if os.path.exists(inps["ESD_dir"]) is False:
        skip = True
        logging.info("ESD directory not found")
        return inps, skip

    else:
        # Identify sub-swaths based on the ESD folder
        ESD_pairs_folders = sorted(glob.glob(os.path.join(inps["ESD_dir"], "2*")))
        if inps["subswath"] == None:
            ESD_pairs_subswath_folders = sorted(
                glob.glob(os.path.join(inps["ESD_dir"], "2*", "IW*"))
            )

            subswath_list = [os.path.basename(i) for i in ESD_pairs_subswath_folders]
            subswath_unique = np.unique(subswath_list)
        else:
            subswath_unique = inps["subswath"]
        for iw in subswath_unique:
            ESD_offset_filename = sorted(
                glob.glob(os.path.join(inps["ESD_dir"], "2*", iw, "combined.off.vrt"))
            )
            if len(ESD_pairs_folders) != len(ESD_offset_filename):
                skip = True
                logging.info(
                    "Skipping. Number of pairs in the ESD folder differs from the number of combined.off.vrt files."
                )
                return inps, skip

            inps["subswath"] = list(subswath_unique)
            logging.info("Sub-swath found {}".format(subswath_unique))
            return inps, skip


def check_coh_points(array_cohPoints):
    """
    Check the number of coherent points in overlapping areas.

    Parameters
    ----------
    array_cohPoints : numpy.ndarray
        Array containing coherence points for overlapping areas.

    Returns
    -------
    skip_plot=True
    """
    skip_plot = False
    # ---------------------#
    # Replace NaN values with 0 to handle missing data
    array_cohPoints = np.nan_to_num(array_cohPoints)

    # Calculate metrics
    min_coh_points = np.min(array_cohPoints, axis=1)
    sum_coh_points = np.sum(array_cohPoints, axis=1)

    # Count pairs with zero coherence in at least one overlapping area
    count_min_zero = np.count_nonzero(min_coh_points == 0)
    # Count pairs with zero coherence across all overlapping areas
    count_sum_zero = np.count_nonzero(sum_coh_points == 0)

    # Log warnings based on coherence checks
    if count_min_zero > 0:
        logging.info(
            "************************************\n"
            "Warning!!! Pairs with low-coherence overlapping areas (coherence < 0.85) have been detected. "
            "A total of {} pairs contain at least one burst overlapping area with low coherence. "
            "This may impact the Enhanced Spectral Diversity estimates for these pairs.\n".format(
                count_min_zero
            )
        )

    if count_sum_zero > 0:
        if count_sum_zero > count_min_zero:
            diff = count_sum_zero - count_min_zero
            logging.info(
                "************************************\n "
                "Warning!!! In addition, {} pairs were found with low coherence in all burst overlapping areas.".format(
                    diff
                )
            )
        elif count_sum_zero < count_min_zero:
            diff = count_min_zero - count_sum_zero
            logging.info(
                "************************************\n "
                "Warning!!! In addition, {} pairs were found with low coherence in all burst overlapping areas.".format(
                    diff
                )
            )
        else:
            logging.info(
                "************************************ \n"
                "Warning!!! The same {} pairs were found with low coherence in all burst overlapping areas.".format(
                    count_min_zero
                )
            )
    if count_min_zero == count_sum_zero and count_sum_zero == array_cohPoints.shape[0]:
        skip_plot = True
    return skip_plot


def prepare_df(array, dateList12, columns, prefix_colName):

    df = pd.DataFrame(array, columns=columns, index=dateList12)

    if prefix_colName == "CohPts_":
        # In case of null values
        df = df.fillna(0)
        df["TotalCohPts"] = df.sum(axis=1)

    # Add prefix and Suffix
    df = df.add_prefix(prefix_colName)
    df = df.add_suffix("_px")
    return df


def modify_df(df):
    # Retrieve other data
    df["RefDate"] = [
        pd.to_datetime(i.split("_")[0], format="%Y%m%d") for i in df.index.tolist()
    ]
    df["RefDate_month"] = df["RefDate"].dt.month

    df["SecDate"] = [
        pd.to_datetime(i.split("_")[1], format="%Y%m%d") for i in df.index.tolist()
    ]
    df["SecDate_month"] = df["SecDate"].dt.month

    df["RefDate_year"] = df["RefDate"].dt.year
    df["SecDate_year"] = df["SecDate"].dt.year

    df["Bt_days"] = (df["SecDate"] - df["RefDate"]).dt.days

    return df


def MAD(x):
    med = np.median(x)
    x = abs(x - med)
    MAD = np.median(x)
    return MAD


def plot_distribution_per_burst_overlap(
    df_stats_medians, df_coh_points, subswath, inps
):

    boxprops = dict(facecolor="lightblue", color="black", linewidth=0.75)
    medianprops = dict(color="red", linewidth=1)
    whiskerprops = dict(color="black", linewidth=0.75)
    capprops = dict(color="black", linewidth=0.75)
    import matplotlib.ticker as mticker

    median_max = np.nanmax(df_stats_medians.iloc[:, :-9])

    fig, axs = plt.subplots(nrows=2, figsize=(8, 15 / 2.54))
    # First boxplot (for medians)
    axs[0].boxplot(
        df_stats_medians.iloc[:, :-9].values,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
    )
    axs[0].set_ylabel("Median Offset Azimuth [px]")
    axs[0].set_ylim(0, np.round(median_max, 2))
    axs[0].set_title("Medians")

    # Second boxplot (for coherent points)
    axs[1].boxplot(
        df_coh_points.iloc[:, :-4].values,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
    )
    axs[1].set_ylabel("#Points Coh >0.85")
    axs[1].set_title("Coherent Points")

    # Rotate x-axis labels
    for ax in axs:
        ax.set_xlabel("Burst Overlapping Area")
        ax.tick_params(axis="x", labelrotation=90)
    axs[1].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    axs[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    # Set the main title and format the figure
    fig.suptitle(
        "Statistics on Each Burst Overlap (Sub-swath {})".format(subswath), fontsize=12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(
        os.path.join(
            inps["inDir"],
            "ESD_azimuth_offsets/",
            "boxplot_stats_at_burst_overlapping_area_{}.png".format(subswath),
        ),
        dpi=300,
    )


# --
def report_pairs(df_stats_medians, inps, subswath):
    # -----------------------
    # Threshold based on
    # --
    # Yagüe-Martinez et al, 2016
    # "Coregistration Accuracy"
    # --
    # Fatttahi et al., 2017
    # "A Network-Based Enhanced Spectral Diversity
    # Approach for TOPS Time-Series Analysis"
    # ------------------

    pairs = (
        df_stats_medians[df_stats_medians["MAD_px"] > threshold].index.to_list().copy()
    )
    out_report = os.path.join(inps["inDir"], "report_pairs_ESD_{}.txt".format(subswath))
    with open(out_report, "w") as fl:
        fl.write(
            "Pairs with Median Absolute Deviation MAD larger than {}\n".format(
                threshold
            )
        )
        fl.write("\n".join(pairs))


def report_dates(df_stats_medians, inps, subswath):

    out_report = os.path.join(
        inps["inDir"], "exclude_dates_ESD_{}.txt".format(subswath)
    )
    # -
    # Create a dataframe
    # Format date, value of the pair
    # -
    datesList = df_stats_medians["RefDate"].values.tolist()
    datesList.extend(df_stats_medians["SecDate"].values.tolist())
    # -
    MADsList = df_stats_medians["MAD_px"].values.tolist()
    MADsList.extend(df_stats_medians["MAD_px"].values.tolist())
    # -
    df2 = pd.DataFrame({"date": datesList, "MAD_px": MADsList})
    # - Calculate median MAD per date
    df3 = df2.groupby("date").median()
    df3 = df3.reset_index()
    # - Exclude
    dates2exclude = df3[df3["date"] > threshold]
    with open(out_report, "w") as fl:
        fl.write(
            "Dates to exclude with MAD (across all pairs) larger than {}\n".format(
                threshold
            )
        )
        fl.write("\n".join(dates2exclude))
    # -


def plot_histograms_of_global_variables(
    df_stats_medians, df_coh_points, subswath, inps
):

    fig, axs = plt.subplots(nrows=2, figsize=(8, 15 / 2.54))
    # First boxplot (for medians)
    values = df_stats_medians["MAD_px"].values.flatten()
    n_bins = 20
    axs[0].hist(values, bins=n_bins)
    axs[0].set_ylabel("Frequency (log-scale)")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("MAD per pair [px]")
    axs[0].set_title("Median Absolute Deviation of Burst Overlap")
    p75 = np.nanpercentile(values, 75)
    axs[0].axvline(p75, c="orange", lw=1, label="75th %ile")
    p90 = np.nanpercentile(values, 90)
    axs[0].axvline(p90, c="red", lw=1, label="90th %ile")
    # axs[0].text(p90, axs[0].get_ylim()[1]/3, '90th %ile', color='red', ha='center', va='center',rotation=90)

    p95 = np.nanpercentile(values, 95)
    axs[0].axvline(p95, c="red", lw=1, ls="--", label="95th %ile")
    # axs[0].text(p95, axs[0].get_ylim()[1]/3, '95th %ile', color='red', ha='center', va='center',rotation=90)

    accuracy_threshold = 0.0009
    axs[0].axvline(accuracy_threshold, c="k", lw=1, label="Accuracy Thresh.")
    axs[0].legend()
    # axs[0].text(accuracy_threshold, axs[0].get_ylim()[1]/3, 'Accuracy Thresh.', color='red', ha='center', va='center',rotation=90)

    # Second boxplot (for coherent points)
    values = df_coh_points["TotalCohPts"].values
    axs[1].hist(values, bins=n_bins)
    axs[1].set_ylabel("Frequency (log-scale)")
    axs[0].set_yscale("log")
    axs[1].set_xlabel("Total of Coherent Points per pair")
    axs[1].set_title("Coherent Points")

    # Set the main title and format the figure
    fig.suptitle(
        "Statistics on Each Burst Overlap (Sub-swath {})".format(subswath), fontsize=12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(
        os.path.join(
            inps["inDir"],
            "ESD_azimuth_offsets/",
            "histograms_stats_at_burst_overlapping_area_{}.png".format(subswath),
        ),
        dpi=300,
    )


def calculate_median_ESD_per_burst(combined_fname):
    """
    Calculates median ESD statistics for each burst overlap.

    Parameters:
    combined_fname (str): Filename of the combined offset file.

    Returns:
    Tuple containing medians, standard deviations, coherent points, and the number of burst overlaps.
    """
    # Load the combined offset, coherence, and interferogram files
    ESD_off = xr.open_dataarray(combined_fname)
    ESD_cor = xr.open_dataarray(combined_fname.replace(".off.vrt", ".cor.vrt"))
    ESD_int = xr.open_dataarray(combined_fname.replace(".off.vrt", ".int.vrt"))

    # Step 1: Mask combined offsets with coherence and interferogram thresholds
    ESD_off = ESD_off.where(ESD_cor > 0.3)
    # Mask as applied on line 91- Script: runESD.py (isce2) (Applied on two moments)
    ESD_off = ESD_off.where(np.abs(ESD_int) > 0)
    ESD_off = ESD_off.squeeze()

    # Retrieve burst overlap coordinates
    max_per_coordinates = ESD_off.max(dim="x")

    # Keep the coordinates were maximum values are different from zero
    coordinates = max_per_coordinates[(max_per_coordinates.notnull())].y.values

    # Group coordinates to find the y-coordinate ranges that separates every
    # burst overlap
    coordinates_split = np.split(coordinates, np.where(np.diff(coordinates) > 1)[0] + 1)

    number_brstOvlp = len(coordinates_split)

    # Step 2: Filter pixels with coherence > 0.85 (ESD threshold)
    # Mask as applied on line 91- Script: runESD.py (isce2)
    ESD_off = ESD_off.where(ESD_cor > 0.85)

    # Calculate median, std, and number of coherent points per burst overlap
    medians = []
    coh_points = []
    for group in coordinates_split:
        medians.append(np.nanmedian(ESD_off.sel(y=group).data))
        coh_points.append(np.count_nonzero(~np.isnan(ESD_off.sel(y=group).data)))

    return medians, coh_points, number_brstOvlp


def calculate_stats_by_subwath(inps, subswath):
    """
    Calculates and saves ESD statistics for each sub-swath.

    Parameters:
        inps (dict): Input directory paths.
        subswath (str): Sub-swath identifier.
    """
    # Find files in the ESD directory
    ESD_offset_filename = sorted(
        glob.glob(os.path.join(inps["ESD_dir"], "2*", subswath, "combined.off.vrt"))
    )
    n_pairs = len(ESD_offset_filename)

    # Define containers
    medians, coh_points, n_brstOvlp = [], [], []

    for fname in ESD_offset_filename:
        # Retrieve statistics
        median_brstOvlp, coh_point_brstOvlp, number_brstOvlp = (
            calculate_median_ESD_per_burst(combined_fname=fname)
        )
        # Save
        medians.extend(median_brstOvlp)
        coh_points.extend(coh_point_brstOvlp)
        n_brstOvlp.append(number_brstOvlp)

    # Check number of burst overlapping areas
    if len(set(n_brstOvlp)) == 1:
        logging.info(
            f"Number of burst overlapping areas found: {n_brstOvlp[0]} ({subswath})"
        )
    else:
        logging.info(
            "Number of burst overlapping areas do not  match among pairs. Error."
        )

    # Reshape lists to arrays
    medians = np.asarray(medians).reshape(n_pairs, n_brstOvlp[0])
    coh_points = np.asarray(coh_points).reshape(n_pairs, n_brstOvlp[0])
    # ------------------------------------#
    skip_processing = check_coh_points(coh_points)

    # Identify ESD pairs format date1_date2
    pairs = [
        os.path.basename(fname.split("/" + subswath)[0])
        for fname in ESD_offset_filename
    ]

    # Coordinates are always read from the first burst overlapping area to the last one
    burst_overlap = ["BstOvlp" + str(i) for i in range(1, n_brstOvlp[0] + 1)]

    if skip_processing == False:
        # --------------------------------------#
        # Prepare dataframes
        # Dataframe of median azimuth offset per burst overlapping areas
        df_stats_medians = prepare_df(
            array=medians,
            columns=burst_overlap,
            dateList12=pairs,
            prefix_colName="MedianAzOff_",
        )

        df_coh_points = prepare_df(
            array=coh_points,
            columns=burst_overlap,
            dateList12=pairs,
            prefix_colName="CohPts_",
        )

        # Transpose to calculate MADs per pair
        df_stats_medians_T = df_stats_medians.T.copy()

        mads = []
        for i in df_stats_medians_T.columns.tolist():
            mads.append(MAD(df_stats_medians_T[i].values))

        # Add MADs across burst overlapping areas to dataframe
        df_stats_medians["MAD_px"] = mads

        # Add temporal parameters
        df_stats_medians = modify_df(df_stats_medians)
        df_coh_points = modify_df(df_coh_points)

        # --------------------------------------------------#
        # Save dataframes
        logging.info("Saving dataframes ...")
        # Define output names
        out_median = os.path.join(
            inps["inDir"],
            "ESD_azimuth_offsets/ESD_azimuth_offset_medians_pairs_{}.csv".format(
                subswath
            ),
        )
        out_coh = os.path.join(
            inps["inDir"],
            "ESD_azimuth_offsets/ESD_azimuth_offset_coh_points_pairs_{}.csv".format(
                subswath
            ),
        )
        df_stats_medians.to_csv(out_median, float_format="%.15f")
        df_coh_points.to_csv(out_coh, float_format="%.15f")

        # --------------------------------------------------#
        # Save summaries from dataframes
        df_stats_medians_describe = df_stats_medians.iloc[:, :-7].describe()
        df_coh_points_describe = df_coh_points.iloc[:, :-7].describe()

        out_median_desc = os.path.join(
            inps["inDir"],
            "ESD_azimuth_offsets/summary_ESD_azimuth_offset_medians_pairs_{}.csv".format(
                subswath
            ),
        )
        out_coh_desc = os.path.join(
            inps["inDir"],
            "ESD_azimuth_offsets/summary_ESD_azimuth_offset_coh_points_pairs_{}.csv".format(
                subswath
            ),
        )

        df_stats_medians_describe.to_csv(out_median_desc, float_format="%.15f")

        df_coh_points_describe.to_csv(out_coh_desc, float_format="%.15f")
        # -----------------------------------------------------#
        # Plot

        logging.info("Plotting figures")

        plot_distribution_per_burst_overlap(
            df_stats_medians, df_coh_points, subswath, inps
        )
        plot_histograms_of_global_variables(
            df_stats_medians, df_coh_points, subswath, inps
        )

        # ----------------------------------------------------#
        # Report
        logging.info("Reporting pairs with large MAD of Azimuth Offset")
        report_pairs(df_stats_medians, inps, subswath)
        report_dates(df_stats_medians, inps, subswath)
    else:
        logging.info(
            "************************************\n"
            "Warning!!! No reports/graphs were made as all ESD pairs have  extreme low coherence (coherence < 0.85).\n"
            "This means that all overlapping areas across all ESD pairs, have NO PIXELS LEFT for the azimuth offset calculation."
            "\n************************************\n"
        )


def run():

    inps = {
        "inDir": os.path.dirname(os.path.abspath(args.inDir)),
        "ESD_dir": os.path.abspath(args.inDir),
        "subswath": args.subswath,
    }
    logging.info("Checking input parameters")
    inps, skip = check_input_directories(inps)
    if skip == False:
        logging.info("Retrieving burst overlap statistics at the sub-swath level")
        for subswath_i in inps["subswath"]:
            calculate_stats_by_subwath(inps, subswath_i)


run()
