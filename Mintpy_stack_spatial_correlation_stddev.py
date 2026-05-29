#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V0.1 May 2026
@author: Bodo Bookhagen
"""

import warnings, argparse, os, logging, sys, tqdm, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
from mintpy.utils import readfile, utils as ut, plot as pp
from mintpy.defaults.plot import *
import pandas as pd
import matplotlib.dates as mdates

# conda install -c conda-forge statsmodels seaborn
# import statsmodels.api as sm
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
import numba as nb
from numba import cuda

# pip install numba-progress
from numba_progress import ProgressBar
import math

register_matplotlib_converters()

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(
    logging.WARNING
)  # Or any other desired level that has a value above that of "INFO".


def tsframe(date, LOS_m, LOS_m_res):
    tsf = pd.DataFrame({"date": date, "LOS_m": LOS_m, "LOS_m_res": LOS_m_res})
    return tsf


def prep_3Draster_forloop(x, maxwindow_size):
    # pads a raster with NaN for faster parallel processing
    # use largest window size to pad to max. size
    nrtimesteps = x.shape[0]
    height = x.shape[1]
    width = x.shape[2]
    window_size_padding = int(np.floor(maxwindow_size / 2))
    x = np.pad(
        x,
        (
            (0, 0),
            (window_size_padding, window_size_padding),
            (window_size_padding, window_size_padding),
        ),
        mode="constant",
        constant_values=np.nan,
    )
    heightp = x.shape[1]
    widthp = x.shape[2]
    pad_height = (maxwindow_size - (heightp % maxwindow_size)) % maxwindow_size
    pad_width = (maxwindow_size - (widthp % maxwindow_size)) % maxwindow_size
    # Apply padding to the DEM to fit into window size
    x = np.pad(
        x,
        (
            (0, 0),
            (0, pad_height),
            (0, pad_width),
        ),
        mode="constant",
        constant_values=np.nan,
    )
    return x, nrtimesteps, height, width


def cmdLineParser():
    parser = argparse.ArgumentParser(
        description="Extract std. dev. with varying radii (windows) and plot spatial correlation."
    )
    parser.add_argument(
        "--npy_out",
        default="data.npy",
        type=str,
        help="Output npy of std. dev. for every time step and window",
        required=True,
    )
    parser.add_argument(
        "--png_out",
        default="timeseries_stddev_windows.png",
        type=str,
        help="Output PNG filename for plot",
        required=True,
    )
    parser.add_argument(
        "--ts1",
        dest="ts1",
        default="geo/geo_timeseries_ERA5_demErr.h5",
        type=str,
        help="Timeseries H5",
        required=True,
    )
    parser.add_argument(
        "--geometry1",
        dest="geometry1",
        default="geo/geo_geometryRadar.h5",
        type=str,
        help="Radar geometry H5",
        required=True,
    )

    return parser.parse_args()


def haversine(lon, lat, dlon, dlat):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    dlon_rad = np.deg2rad(dlon)
    dlat_rad = np.deg2rad(dlat)

    # haversine formula
    a = (
        np.sin(dlat_rad / 2) ** 2
        + np.cos(lat_rad) * np.cos(lat_rad) * np.sin(dlon_rad / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r * 1000


def calc_stddev_raster(raster, window_size):
    stddev_results = np.empty((raster.shape[0], len(window_size), 4), dtype=np.float32)
    stddev_results.fill(np.nan)
    stddev_grid = np.empty((raster.shape[1], raster.shape[2]), dtype=np.float32)
    stddev_grid.fill(np.nan)
    for l in tqdm.tqdm(range(raster.shape[0])):
        for k in range(len(window_size)):
            window_size_half = int(window_size[k] / 2)
            for i in range(window_size_half, raster.shape[1] - window_size_half):
                for j in range(window_size_half, raster.shape[2] - window_size_half):
                    z = raster[
                        l,
                        i - window_size_half : i + window_size_half + 1,
                        j - window_size_half : j + window_size_half + 1,
                    ].ravel()
                    stddev_grid[i, j] = np.std(z)
            stddev_results[l, k, 0] = np.nanmean(stddev_grid)
            stddev_results[l, k, 1] = np.nanstd(stddev_grid)
            stddev_results[l, k, 2] = np.nanpercentile(
                stddev_grid, 75
            ) - np.nanpercentile(stddev_grid, 25)
            stddev_results[l, k, 3] = np.nanmedian(stddev_grid)
    return stddev_results


@nb.njit(parallel=True, fastmath=True)
def local_std_multiwindow(image, window_sizes):
    """
    Compute per-pixel standard deviation for multiple window sizes.

    Parameters
    ----------
    image : 2D numpy array (float32/float64)
        Input grayscale image.
    window_sizes : 1D numpy array of odd integers
        Example: np.array([3, 5, 7])

    Returns
    -------
    result : 3D numpy array
        Shape = (num_windows, height, width)
        result[k, y, x] = std deviation at pixel (y,x)
        using window_sizes[k]
    """
    h, w = image.shape
    n_windows = len(window_sizes)
    result = np.zeros((n_windows, h, w), dtype=np.float32)
    for k in nb.prange(n_windows):
        ws = window_sizes[k]
        r = ws // 2
        for y in range(h):
            y0 = max(0, y - r)
            y1 = min(h, y + r + 1)
            for x in range(w):
                x0 = max(0, x - r)
                x1 = min(w, x + r + 1)

                # First pass: mean
                s = 0.0
                count = 0
                for yy in range(y0, y1):
                    for xx in range(x0, x1):
                        s += image[yy, xx]
                        count += 1
                mean = s / count

                # Second pass: variance
                var = 0.0
                for yy in range(y0, y1):
                    for xx in range(x0, x1):
                        d = image[yy, xx] - mean
                        var += d * d

                result[k, y, x] = np.sqrt(var / count)

    return result


@nb.jit(nopython=True, fastmath=True, parallel=True, nogil=True)
def calc_stddev_raster_numba(totaliterations, raster, window_size, progress_proxy):
    stddev_results = np.empty((len(window_size), raster.shape[0], 4), dtype=np.float32)
    stddev_results.fill(np.nan)
    stddev_grid = np.empty(
        (raster.shape[0], raster.shape[1], raster.shape[2]), dtype=np.float32
    )
    stddev_grid.fill(np.nan)
    for l in nb.prange(len(window_size)):
        window_size_half = int(window_size[l] / 2)
        for k in nb.prange(raster.shape[0]):
            for i in nb.prange(window_size_half, raster.shape[1] - window_size_half):
                for j in nb.prange(
                    window_size_half, raster.shape[2] - window_size_half
                ):
                    z = raster[
                        k,
                        i - window_size_half : i + window_size_half + 1,
                        j - window_size_half : j + window_size_half + 1,
                    ].ravel()
                    stddev_grid[k, i, j] = np.std(z)
            stddev_results[l, k, 0] = np.nanmean(stddev_grid)
            stddev_results[l, k, 1] = np.nanstd(stddev_grid)
            stddev_results[l, k, 2] = np.nanpercentile(
                stddev_grid, 75
            ) - np.nanpercentile(stddev_grid, 25)
            stddev_results[l, k, 3] = np.nanmedian(stddev_grid)
        progress_proxy.update(1)
    return stddev_results


@cuda.jit
def local_std_cuda(image, window_sizes, output):
    """
    Parameters
    ----------
    image : 2D float32 array
    window_sizes : 1D int32 array
    output : 3D float32 array
        Shape: (num_windows, height, width)
    """
    # 3D grid:
    # x -> image width
    # y -> image height
    # z -> window index

    x, y, k = cuda.grid(3)
    h = image.shape[0]
    w = image.shape[1]
    if x >= w or y >= h or k >= window_sizes.shape[0]:
        return

    ws = window_sizes[k]
    r = ws // 2

    # ------------------------------------------
    # Compute local bounds
    # ------------------------------------------
    x0 = max(0, x - r)
    x1 = min(w, x + r + 1)
    y0 = max(0, y - r)
    y1 = min(h, y + r + 1)

    # ------------------------------------------
    # First pass: mean
    # ------------------------------------------
    s = 0.0
    count = 0
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            s += image[yy, xx]
            count += 1

    mean = s / count

    # ------------------------------------------
    # Second pass: variance
    # ------------------------------------------
    var = 0.0
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            d = image[yy, xx] - mean
            var += d * d

    output[k, y, x] = math.sqrt(var / count)


def compute_local_std_cuda(image, window_sizes):
    """
    Compute local std deviation maps on GPU.

    Parameters
    ----------
    image : 2D numpy array
        float32 recommended
    window_sizes : list or ndarray
        odd integers

    Returns
    -------
    output : 3D numpy array
        Shape = (num_windows, H, W)
    """

    image = image.astype(np.float32)
    window_sizes = np.asarray(window_sizes, dtype=np.int32)

    h, w = image.shape
    n_windows = len(window_sizes)
    output = np.zeros((n_windows, h, w), dtype=np.float32)

    d_image = cuda.to_device(image)
    d_windows = cuda.to_device(window_sizes)
    d_output = cuda.to_device(output)

    threadsperblock = (16, 16, 1)
    blockspergrid_x = math.ceil(w / threadsperblock[0])
    blockspergrid_y = math.ceil(h / threadsperblock[1])
    blockspergrid_z = n_windows

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    local_std_cuda[blockspergrid, threadsperblock](d_image, d_windows, d_output)

    return d_output.copy_to_host()


def plot_window_sizes(pngfn):
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(12, 8), dpi=300, layout="constrained"
    )
    # plot all time steps - need to do color cycles
    vcolor = plt.cm.viridis(np.linspace(0, 1, result.shape[0]))
    for i in range(1, result.shape[0]):
        ax1.plot(
            window_sizes_m / 1000, result[i, :, 0] * 1000, "-", lw=0.4, color=vcolor[i]
        )
    ax1.grid()
    # ax1.set_xlabel("Window Size (km)", fontsize=12)
    ax1.set_ylabel("Averaged Std. Dev.\nfrom each time step (mm)", fontsize=12)
    ax1.set_ylim([0, 40])
    ax2.plot(
        window_sizes_m / 1000,
        np.mean(result[:, :, 0], axis=0) * 1000,
        lw=2,
        color="navy",
    )
    ax2.plot(
        window_sizes_m / 1000,
        np.mean(result[:, :, 0], axis=0) * 1000
        + np.std(result[:, :, 0], axis=0) * 1000,
        lw=0.1,
        color="navy",
    )
    ax2.plot(
        window_sizes_m / 1000,
        np.mean(result[:, :, 0], axis=0) * 1000
        - np.std(result[:, :, 0], axis=0) * 1000,
        lw=0.1,
        color="navy",
    )
    ax2.fill_between(
        window_sizes_m / 1000,
        np.mean(result[:, :, 0], axis=0) * 1000
        + np.std(result[:, :, 0], axis=0) * 1000,
        np.mean(result[:, :, 0], axis=0) * 1000
        - np.std(result[:, :, 0], axis=0) * 1000,
        color="navy",
        alpha=0.1,
    )
    ax2.set_xlabel("Window Size (km)", fontsize=12)
    ax2.set_ylabel("Averaged Std. Dev.\nfor all time steps (mm)", fontsize=12)
    ax2.grid()
    ax2.set_ylim([0, 40])
    fig.savefig(pngfn, dpi=300)
    plt.close()


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    warnings.simplefilter("ignore")

    args = cmdLineParser()
    # testing purposes:
    # parser = argparse.ArgumentParser(
    #     description="Extract ts from coordinate and store as pandas dataframe. Use a 3x3 window and plot simple linear regression."
    # )
    # args = parser.parse_args()
    # args.vel1 = "/raid2-manaslu/sofia/01_proyect/s1/asc_t76/15_Cafayate_Salta_Area_Geometry/02_stacks_Cafayate/01_DenseStack/mintpy_T76Cafayate/mintpy_rg20az4/05_mintpy_nr3_rg20az4_T76/01_fullTs_BW3/geo/geo_velocity_raw.h5 "
    # args.ts1 = "/raid2-manaslu/sofia/01_proyect/s1/asc_t76/15_Cafayate_Salta_Area_Geometry/02_stacks_Cafayate/01_DenseStack/mintpy_T76Cafayate/mintpy_rg20az4/05_mintpy_nr3_rg20az4_T76/01_fullTs_BW3_exclude_pj/geo/geo_timeseries.h5"
    # args.geometry1 = "/raid2-manaslu/sofia/01_proyect/s1/asc_t76/15_Cafayate_Salta_Area_Geometry/02_stacks_Cafayate/01_DenseStack/mintpy_T76Cafayate/mintpy_rg20az4/05_mintpy_nr3_rg20az4_T76/inputs/geo_geometryRadar.h5"
    # args.npy_out = "01_fullTs_BW3_spatial_correlation_stddev_timeseries.npy"
    # args.png_out = "01_fullTs_BW3_spatial_correlation_stddev_timeseries.png"

    incAngle, atr = ut.readfile.read(args.geometry1, datasetName="incidenceAngle")
    height, atr = ut.readfile.read(args.geometry1, datasetName="height")
    raster, atr = ut.readfile.read(args.ts1, datasetName="timeseries")
    # get center of image coordinate and dates
    dates_ts1, raster1, los_sd_ts1 = ut.read_timeseries_yx(
        y=np.int16(raster.shape[1] / 2),
        x=np.int16(raster.shape[2] / 2),
        ts_file=args.ts1,
        win_size=3,
        unit="m",
    )
    lon = np.float32(atr["X_FIRST"])
    lat = np.float32(atr["Y_FIRST"])
    dlon = np.abs(np.float32(atr["X_STEP"]))
    dlat = np.abs(np.float32(atr["Y_STEP"]))
    grid_spatial_resolution_m = haversine(lon, lat, dlon, dlat)
    window_sizes_m = np.array(
        [
            500,
            1000,
            2500,
            5000,
            7500,
            10000,
            12500,
            15000,
            17500,
            20000,
            22500,
            25000,
            25750,
            30000,
            32500,
            35000,
            37500,
            40000,
            42500,
            45000,
            47500,
            50000,
            52500,
            55000,
        ]
    )
    window_sizes = np.int16(np.ceil(window_sizes_m / grid_spatial_resolution_m))
    np.save("window_sizes_m.npy", window_sizes_m)
    np.save("window_sizes.npy", window_sizes)
    np.save("dates_ts1.npy", dates_ts1)

    logging.info(
        "Padding LOS timeseries with max. window size of %d" % max(window_sizes)
    )
    raster, nrtimesteps, height, width = prep_3Draster_forloop(
        raster, max(window_sizes)
    )
    # Multi-CPU implementation is slow:
    # start = time.time()
    # stddev_results = calc_stddev_raster(
    #     raster,
    #     window_size,
    # )
    # stop = time.time()
    # logging.info(
    #     "numpy std. dev. calculation for timeseries took %2.2f seconds or %2.2f minutes"
    #     % (stop - start, (stop - start) / 60)
    # )
    #
    # Multi-CPU implementation is slow:
    # totaliterations = raster.shape[0]
    # logging.info(
    #     "Iterate through all timesteps and all pixels: %s" % f"{totaliterations:,}"
    # )
    # start = time.time()
    # with ProgressBar(total=totaliterations) as progress:
    #     stddev_results = calc_stddev_raster_numba(
    #         totaliterations, raster, window_size, progress
    #     )
    # stop = time.time()
    # logging.info(
    #     "numba std. dev. calculation for timeseries took %2.2f seconds or %2.2f minutes"
    #     % (stop - start, (stop - start) / 60)
    # )

    # logging.info("Run every time step separately on CUDA and calculate all windows")
    # result = np.empty((raster.shape[0], len(window_sizes), 4), dtype=np.float32)
    # result.fill(np.nan)
    # for i in tqdm.tqdm(range(raster.shape[0]), desc="Time Step"):
    #     std_maps = compute_local_std_cuda(raster[i, :, :], window_sizes)
    #     for j in range(len(window_sizes)):
    #         result[i, j, 0] = np.nanmean(std_maps[j, :, :])
    #         result[i, j, 1] = np.nanstd(std_maps[j, :, :])
    #         result[i, j, 2] = np.nanpercentile(
    #             std_maps[j, :, :], 75
    #         ) - np.nanpercentile(std_maps[j, :, :], 25)
    #         result[i, j, 3] = np.nanmedian(std_maps[j, :, :])
    # # result is Time x window_length x statistics
    # # arrfn = "01_fullTs_BW3_geo_timeseries_windowsizes_%d.npy" % (len(window_sizes))
    # np.save(args.npy_out, result)
    #
    nr_steps = 100
    logging.info(
        "Run only %d time step separately on CUDA and calculate all windows" % nr_steps
    )
    raster0_steps = np.int16(np.linspace(1, raster.shape[0] - 1, nr_steps))
    result = np.empty((nr_steps, len(window_sizes), 4), dtype=np.float32)
    result.fill(np.nan)
    for i in tqdm.tqdm(range(nr_steps), desc="Time Step"):
        std_maps = compute_local_std_cuda(raster[raster0_steps[i], :, :], window_sizes)
        for j in range(len(window_sizes)):
            result[i, j, 0] = np.nanmean(std_maps[j, :, :])
            result[i, j, 1] = np.nanstd(std_maps[j, :, :])
            result[i, j, 2] = np.nanpercentile(
                std_maps[j, :, :], 75
            ) - np.nanpercentile(std_maps[j, :, :], 25)
            result[i, j, 3] = np.nanmedian(std_maps[j, :, :])
    # result is Time x window_length x statistics
    np.save(args.npy_out, result)

    plot_window_sizes(args.png_out)
