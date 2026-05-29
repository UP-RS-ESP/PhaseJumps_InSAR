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
import pandas as pd
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


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


def plot_all_timesteps_window_sizes(pngfn):
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=(12, 8), dpi=300, layout="constrained"
    )
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    for i in range(1, ts_spatialcorr_raw.shape[0]):
        ax1.plot(
            window_sizes_m / 1000,
            ts_spatialcorr_raw[i, :, 0] * 1000,
            "-",
            lw=0.4,
            color="gray",
            label="uncorrected",
        )
    ax1.plot(
        window_sizes_m / 1000,
        np.nanmean(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000,
        "-",
        lw=2,
        color="k",
        label="mean uncorrected",
    )
    ax1.grid()
    # ax1.legend()
    ax1.set_ylabel("Averaged Std. Dev.\n (mm)", fontsize=14)
    ax1.set_ylim([0, 40])
    # ax1.set_title("Uncorrected", fontsize=16)
    ax1.annotate(
        "A: Uncorrected time series",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(+0.5, -0.5),
        textcoords="offset fontsize",
        fontsize=18,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="black", pad=3.0),
    )
    for i in range(1, ts_spatialcorr_IonSB.shape[0]):
        ax2.plot(
            window_sizes_m / 1000,
            ts_spatialcorr_IonSB[i, :, 0] * 1000,
            "-",
            lw=0.4,
            color="gray",
            label="IonSB",
        )
    ax2.plot(
        window_sizes_m / 1000,
        np.nanmean(ts_spatialcorr_IonSB[:, :, 0], axis=0) * 1000,
        "-",
        lw=2,
        color="k",
        label="mean IonoSB",
    )
    ax2.grid()
    # ax2.legend()
    ax2.set_ylabel("Averaged Std. Dev.\n (mm)", fontsize=14)
    ax2.set_ylim([0, 40])
    # ax2.set_title("Split-Beam ionospheric correction", fontsize=16)
    ax2.annotate(
        "B: Split-spectrum ionospheric corrected time series",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(+0.5, -0.5),
        textcoords="offset fontsize",
        fontsize=18,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="black", pad=3.0),
    )
    for i in range(1, ts_spatialcorr_IonSB_ERA5wet.shape[0]):
        ax3.plot(
            window_sizes_m / 1000,
            ts_spatialcorr_IonSB_ERA5wet[i, :, 0] * 1000,
            "-",
            lw=0.4,
            color="gray",
            label="IonSB_ERA5wet",
        )
    ax3.plot(
        window_sizes_m / 1000,
        np.nanmean(ts_spatialcorr_IonSB_ERA5wet[:, :, 0], axis=0) * 1000,
        "-",
        lw=2,
        color="k",
        label="mean IonoSB-ERA5wet",
    )
    ax3.grid()
    # ax3.legend()
    ax3.set_xlabel("Window Size (km)", fontsize=14)
    ax3.set_ylabel("Averaged Std. Dev.\n (mm)", fontsize=14)
    ax3.set_ylim([0, 40])
    # ax3.set_title("Split-Beam ionospheric and ERA5-wet correction", fontsize=16)
    ax3.annotate(
        "C: Split-spectrum ionospheric and ERA5-wet corrected time series",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(+0.5, -0.5),
        textcoords="offset fontsize",
        fontsize=18,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="black", pad=3.0),
    )
    fig.savefig(pngfn, dpi=300)
    plt.close()


def plot_all_timesteps_window_sizes_colorlines(pngfn):
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=(12, 8), dpi=300, layout="constrained"
    )
    vcolor = plt.cm.viridis(np.linspace(0, 1, ts_spatialcorr_raw.shape[0]))
    for i in range(1, ts_spatialcorr_raw.shape[0]):
        ax1.plot(
            window_sizes_m / 1000,
            ts_spatialcorr_raw[i, :, 0] * 1000,
            "-",
            lw=0.7,
            color=vcolor[i],
            label="uncorrected",
        )
    ax1.grid()
    # ax1.legend()
    ax1.set_ylabel("Averaged Std. Dev.\n(uncorrected)", fontsize=12)
    ax1.set_ylim([0, 40])
    ax1.set_title("Uncorrected")
    for i in range(1, ts_spatialcorr_IonSB.shape[0]):
        ax2.plot(
            window_sizes_m / 1000,
            ts_spatialcorr_IonSB[i, :, 0] * 1000,
            "-",
            lw=0.7,
            color=vcolor[i],
            label="IonSB",
        )
    ax2.grid()
    # ax2.legend()
    ax2.set_ylabel("Averaged Std. Dev.\n(IonSB)", fontsize=12)
    ax2.set_ylim([0, 40])
    ax2.set_title("IonoSB")
    for i in range(1, ts_spatialcorr_IonSB_ERA5wet_dry.shape[0]):
        ax3.plot(
            window_sizes_m / 1000,
            ts_spatialcorr_IonSB_ERA5wet_dry[i, :, 0] * 1000,
            "-",
            lw=0.6,
            color=vcolor[i],
            label="IonSB_ERA5wet_dry",
        )
    ax3.grid()
    # ax3.legend()
    ax3.set_xlabel("Window Size (km)", fontsize=12)
    ax3.set_ylabel("Averaged Std. Dev.\n(IonSB_ERA5wet_dry)", fontsize=12)
    ax3.set_ylim([0, 40])
    ax3.set_title("IonSB_ERA5wet_dry")
    fig.savefig(pngfn, dpi=300)
    plt.close()


def plot_all_timesteps_difference_window_sizes(pngfn):
    fig, (ax1, ax4) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(12, 8), dpi=300, layout="constrained"
    )
    ax1.plot(
        window_sizes_m / 1000,
        ts_spatialcorr_IonSB[1, :, 0] * 1000 - ts_spatialcorr_raw[1, :, 0] * 1000,
        "-",
        lw=0.4,
        color="gray",
        label="individual date $\Delta$",
    )
    for i in range(2, ts_spatialcorr_raw.shape[0]):
        ax1.plot(
            window_sizes_m / 1000,
            ts_spatialcorr_IonSB[i, :, 0] * 1000 - ts_spatialcorr_raw[i, :, 0] * 1000,
            "-",
            lw=0.4,
            color="gray",
        )
    ax1.grid()
    ax1.plot(
        window_sizes_m / 1000,
        np.nanmean(
            ts_spatialcorr_IonSB[:, :, 0] * 1000 - ts_spatialcorr_raw[:, :, 0] * 1000,
            axis=0,
        ),
        "-",
        lw=2,
        label="Mean $\Delta$",
        color="k",
    )
    ax1.legend(fontsize=12)
    ax1.set_ylabel(r"$\Delta$ Averaged Std. Dev. (mm)", fontsize=14)
    ax1.tick_params(axis="both", which="major", labelsize=14)
    ax1.set_ylim([-3, 3])
    # ax1.set_title(
    #     "Uncorrected $\it{minus}$ Split-Beam Ionospheric Correction", fontsize=16
    # )
    ax1.annotate(
        "A: Split-spectrum ionospheric correction $\it{minus}$\nuncorrected time series",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(+0.5, -0.5),
        textcoords="offset fontsize",
        fontsize=18,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="black", pad=3.0),
    )
    # for i in range(1, ts_spatialcorr_IonSB.shape[0]):
    #     ax2.plot(
    #         window_sizes_m / 1000,
    #         ts_spatialcorr_IonSB[i, :, 0] * 1000
    #         - ts_spatialcorr_IonSB_ERA5wet[i, :, 0] * 1000,
    #         "-",
    #         lw=0.4,
    #         color="gray",
    #         label="IonSB - IonSB_ERA5wet_dry",
    #     )
    # ax2.grid()
    # ax2.plot(
    #     window_sizes_m / 1000,
    #     np.nanmean(
    #         ts_spatialcorr_IonSB[:, :, 0] * 1000
    #         - ts_spatialcorr_IonSB_ERA5wet[:, :, 0] * 1000,
    #         axis=0,
    #     ),
    #     "-",
    #     lw=2,
    #     color="k",
    # )
    # # ax2.legend()
    # ax2.set_ylabel(r"$\Delta$ Averaged Std. Dev.", fontsize=14)
    # ax2.set_ylim([-20, 20])
    # ax2.set_title(
    #     "Split-Beam Ionospheric - Split-Beam Ionospheric plus ERA5-wet Correction",
    #     fontsize=16,
    # )
    # for i in range(1, ts_spatialcorr_IonSB_ERA5wet.shape[0]):
    #     ax3.plot(
    #         window_sizes_m / 1000,
    #         ts_spatialcorr_IonSB_ERA5wet[i, :, 0] * 1000
    #         - ts_spatialcorr_IonSB_ERA5wet_dry[i, :, 0] * 1000,
    #         "-",
    #         lw=0.4,
    #         color="gray",
    #         label="IonSB_ERA5wet_dry - IonSB_ERA5wet_dry_demErr",
    #     )
    # ax3.grid()
    # ax3.plot(
    #     window_sizes_m / 1000,
    #     np.nanmean(
    #         ts_spatialcorr_IonSB_ERA5wet[:, :, 0] * 1000
    #         - ts_spatialcorr_IonSB_ERA5wet_dry[:, :, 0] * 1000,
    #         axis=0,
    #     ),
    #     "-",
    #     lw=2,
    #     color="k",
    # )
    # # ax3.legend()
    # ax3.set_xlabel("Window Size (km)", fontsize=14)
    # ax3.set_ylabel(r"$\Delta$ Averaged Std. Dev.", fontsize=14)
    # ax3.set_ylim([-5, 5])
    # ax3.set_title("ERA5-wet - ERA5-wet+dry Correction", fontsize=16)
    ax4.plot(
        window_sizes_m / 1000,
        ts_spatialcorr_IonSB_ERA5wet_dry[1, :, 0] * 1000
        - ts_spatialcorr_IonSB[1, :, 0] * 1000,
        "-",
        lw=0.4,
        color="gray",
        label="individual date $\Delta$",
    )
    for i in range(2, ts_spatialcorr_IonSB_ERA5wet.shape[0]):
        ax4.plot(
            window_sizes_m / 1000,
            ts_spatialcorr_IonSB_ERA5wet_dry[i, :, 0] * 1000
            - ts_spatialcorr_IonSB[i, :, 0] * 1000,
            "-",
            lw=0.4,
            color="gray",
        )
    ax4.grid()
    ax4.plot(
        window_sizes_m / 1000,
        np.nanmean(
            ts_spatialcorr_IonSB_ERA5wet_dry[:, :, 0] * 1000
            - ts_spatialcorr_IonSB[:, :, 0] * 1000,
            axis=0,
        ),
        "-",
        lw=2,
        color="k",
        label="Mean $\Delta$",
    )
    ax4.legend(fontsize=12)
    ax4.set_xlabel("Window Size (km)", fontsize=14)
    ax4.set_ylabel(r"$\Delta$ Averaged Std. Dev. (mm)", fontsize=14)
    ax4.set_ylim([-20, 20])
    ax4.tick_params(axis="both", which="major", labelsize=12)
    # ax4.set_title(
    #     "Split-Beam Ionospheric $\it{minus}$ ERA5-wet+dry Correction", fontsize=16
    # )
    ax4.annotate(
        "B: Split-spectrum ionospheric and ERA5-wet+dry $\it{minus}$\nsplit-beam ionospheric correction time series",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(+0.5, -0.5),
        textcoords="offset fontsize",
        fontsize=18,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="black", pad=3.0),
    )
    fig.savefig(pngfn, dpi=300)
    plt.close()


def plot_all_timesteps_difference_window_sizes_colors(pngfn):
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=(12, 8), dpi=300, layout="constrained"
    )
    vcolor = plt.cm.viridis(np.linspace(0, 1, ts_spatialcorr_raw.shape[0]))
    for i in range(1, ts_spatialcorr_raw.shape[0]):
        ax1.plot(
            window_sizes_m / 1000,
            ts_spatialcorr_raw[i, :, 0] * 1000 - ts_spatialcorr_IonSB[i, :, 0] * 1000,
            "-",
            lw=0.7,
            color=vcolor[i],
            label="uncorrected - IonoSB",
        )
    ax1.grid()
    # ax1.legend()
    ax1.set_ylabel("$\Delta$ Averaged Std. Dev.", fontsize=12)
    ax1.set_ylim([-10, 10])
    ax1.set_title("Uncorrected - IonSB")
    for i in range(1, ts_spatialcorr_IonSB.shape[0]):
        ax2.plot(
            window_sizes_m / 1000,
            ts_spatialcorr_IonSB[i, :, 0] * 1000
            - ts_spatialcorr_IonSB_ERA5wet_dry[i, :, 0] * 1000,
            "-",
            lw=0.7,
            color=vcolor[i],
            label="IonSB - IonSB_ERA5wet_dry",
        )
    ax2.grid()
    # ax2.legend()
    ax2.set_ylabel("$\Delta$ Averaged Std. Dev.", fontsize=12)
    ax2.set_ylim([-10, 10])
    ax2.set_title("IonoSB - IonSB_ERA5wet_dry")
    for i in range(1, ts_spatialcorr_IonSB_ERA5wet_dry.shape[0]):
        ax3.plot(
            window_sizes_m / 1000,
            ts_spatialcorr_IonSB_ERA5wet_dry[i, :, 0] * 1000
            - ts_spatialcorr_IonSB_ERA5wet_dry_demErr[i, :, 0] * 1000,
            "-",
            lw=0.6,
            color=vcolor[i],
            label="IonSB_ERA5wet_dry - IonSB_ERA5wet_dry_demErr",
        )
    ax3.grid()
    # ax3.legend()
    ax3.set_xlabel("Window Size (km)", fontsize=12)
    ax3.set_ylabel("$\Delta$ Averaged Std. Dev.", fontsize=12)
    ax3.set_ylim([-10, 10])
    ax3.set_title("IonSB_ERA5wet_dry - IonSB_ERA5wet_dry_demErr")
    fig.savefig(pngfn, dpi=300)
    plt.close()


def plot_window_sizes(pngfn):
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=(12, 8), dpi=300, layout="constrained"
    )
    # plot 10th time step
    # vcolor = plt.cm.viridis(np.linspace(0, 1, result.shape[0]))
    i = 5
    ax1.plot(
        window_sizes_m / 1000,
        ts_spatialcorr_raw[i, :, 0] * 1000,
        "-",
        lw=0.5,
        color="k",
        label="uncorrected",
    )
    ax1.plot(
        window_sizes_m / 1000,
        ts_spatialcorr_IonSB[i, :, 0] * 1000,
        "-",
        lw=0.5,
        color="darkorange",
        label="IonSB",
    )
    ax1.plot(
        window_sizes_m / 1000,
        ts_spatialcorr_IonSB_ERA5wet[i, :, 0] * 1000,
        "-",
        lw=0.5,
        color="navy",
        label="IonSB_ERA5wet",
    )
    ax1.plot(
        window_sizes_m / 1000,
        ts_spatialcorr_IonSB_ERA5wet_dry[i, :, 0] * 1000,
        "-",
        lw=0.5,
        color="lightblue",
        label="IonSB_ERA5wet_dry",
    )
    ax1.plot(
        window_sizes_m / 1000,
        ts_spatialcorr_IonSB_ERA5wet_dry_demErr[i, :, 0] * 1000,
        "-",
        lw=0.5,
        color="purple",
        label="IonSB_ERA5wet_dry_demErr",
    )
    ax1.grid()
    ax1.legend()
    # ax1.set_xlabel("Window Size (km)", fontsize=12)
    ax1.set_ylabel("Averaged Std. Dev.\nfrom one time step (mm)", fontsize=12)
    ax1.set_ylim([0, 40])
    ax1.set_title("Std. Dev. for timestep %d" % i)
    ax2.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000,
        lw=2,
        color="k",
        label="uncorrected",
    )
    ax2.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_IonSB[:, :, 0], axis=0) * 1000,
        lw=2,
        color="darkorange",
        label="IonSB",
    )
    ax2.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_IonSB_ERA5wet[:, :, 0], axis=0) * 1000,
        lw=2,
        color="navy",
        label="IonSB_ERA5wet",
    )
    ax2.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_IonSB_ERA5wet_dry[:, :, 0], axis=0) * 1000,
        lw=2,
        color="lightblue",
        label="IonSB_ERA5wet_dry",
    )
    ax2.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_IonSB_ERA5wet_dry_demErr[:, :, 0], axis=0) * 1000,
        lw=2,
        color="purple",
        label="IonSB_ERA5wet_dry_demErr",
    )
    ax2.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000
        + np.std(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000,
        lw=0.1,
        color="k",
    )
    ax2.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000
        - np.std(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000,
        lw=0.1,
        color="k",
    )
    ax2.fill_between(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000
        + np.std(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000,
        np.mean(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000
        - np.std(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000,
        color="k",
        alpha=0.05,
    )
    ax2.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_IonSB_ERA5wet_dry[:, :, 0], axis=0) * 1000
        + np.std(ts_spatialcorr_IonSB_ERA5wet_dry[:, :, 0], axis=0) * 1000,
        lw=0.2,
        color="lightblue",
    )
    ax2.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_IonSB_ERA5wet_dry[:, :, 0], axis=0) * 1000
        - np.std(ts_spatialcorr_IonSB_ERA5wet_dry[:, :, 0], axis=0) * 1000,
        lw=0.2,
        color="lightblue",
    )
    ax2.fill_between(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_IonSB_ERA5wet_dry[:, :, 0], axis=0) * 1000
        + np.std(ts_spatialcorr_IonSB_ERA5wet_dry[:, :, 0], axis=0) * 1000,
        np.mean(ts_spatialcorr_IonSB_ERA5wet_dry[:, :, 0], axis=0) * 1000
        - np.std(ts_spatialcorr_IonSB_ERA5wet_dry[:, :, 0], axis=0) * 1000,
        color="lightblue",
        alpha=0.3,
    )
    ax2.set_xlabel("Window Size (km)", fontsize=12)
    ax2.set_ylabel("Averaged Std. Dev.\nfor all time steps (mm)", fontsize=12)
    ax2.grid()
    ax2.set_ylim([0, 40])
    ax2.legend()
    ax2.set_title(
        "Averaged Std. Dev. for all time steps (n=%d) - Raw and ERA5-wet+dry"
        % ts_spatialcorr_IonSB.shape[0]
    )
    ax3.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000,
        lw=2,
        color="k",
        label="uncorrected",
    )
    ax3.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_IonSB[:, :, 0], axis=0) * 1000,
        lw=2,
        color="darkorange",
        label="IonSB",
    )
    ax3.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_IonSB_ERA5wet[:, :, 0], axis=0) * 1000,
        lw=2,
        color="navy",
        label="IonSB_ERA5wet",
    )
    ax3.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_IonSB_ERA5wet_dry[:, :, 0], axis=0) * 1000,
        lw=2,
        color="lightblue",
        label="IonSB_ERA5wet_dry",
    )
    ax3.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_IonSB_ERA5wet_dry_demErr[:, :, 0], axis=0) * 1000,
        lw=2,
        color="purple",
        label="IonSB_ERA5wet_dry_demErr",
    )
    ax3.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000
        + np.std(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000,
        lw=0.1,
        color="k",
    )
    ax3.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000
        - np.std(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000,
        lw=0.1,
        color="k",
    )
    ax3.fill_between(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000
        + np.std(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000,
        np.mean(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000
        - np.std(ts_spatialcorr_raw[:, :, 0], axis=0) * 1000,
        color="k",
        alpha=0.05,
    )
    ax3.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_IonSB[:, :, 0], axis=0) * 1000
        + np.std(ts_spatialcorr_IonSB[:, :, 0], axis=0) * 1000,
        lw=0.2,
        color="darkorange",
    )
    ax3.plot(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_IonSB[:, :, 0], axis=0) * 1000
        - np.std(ts_spatialcorr_IonSB[:, :, 0], axis=0) * 1000,
        lw=0.2,
        color="darkorange",
    )
    ax3.fill_between(
        window_sizes_m / 1000,
        np.mean(ts_spatialcorr_IonSB[:, :, 0], axis=0) * 1000
        + np.std(ts_spatialcorr_IonSB[:, :, 0], axis=0) * 1000,
        np.mean(ts_spatialcorr_IonSB[:, :, 0], axis=0) * 1000
        - np.std(ts_spatialcorr_IonSB[:, :, 0], axis=0) * 1000,
        color="darkorange",
        alpha=0.3,
    )
    ax3.set_xlabel("Window Size (km)", fontsize=12)
    ax3.set_ylabel(
        "Averaged Std. Dev.\nfor all time steps (mm) - Raw and IonSB", fontsize=12
    )
    ax3.grid()
    ax3.set_ylim([0, 40])
    ax3.legend()
    ax3.set_title(
        "Averaged Std. Dev. for all time steps (n=%d)" % ts_spatialcorr_IonSB.shape[0]
    )
    fig.savefig(pngfn, dpi=300)
    plt.close()


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    warnings.simplefilter("ignore")

    args = cmdLineParser()
    # testing purposes:
    parser = argparse.ArgumentParser(
        description="Extract ts from coordinate and store as pandas dataframe. Use a 3x3 window and plot simple linear regression."
    )
    args = parser.parse_args()
    args.png_out = "01_fullTs_BW3_exclude_pj_spatial_correlation_stddev_timeseries_n100_comparison.png"
    args.png_out2 = "01_fullTs_BW3_exclude_pj_spatial_correlation_stddev_timeseries_n100_comparison_alltimesteps.png"
    args.png_out3 = "01_fullTs_BW3_exclude_pj_spatial_correlation_stddev_timeseries_n100_comparison_alltimesteps_difference.png"
    args.ts_1 = (
        "01_fullTs_BW3_exclude_pj_spatial_correlation_stddev_timeseries_n100.npy"
    )
    args.ts_2 = (
        "01_fullTs_BW3_exclude_pj_IonSB_spatial_correlation_stddev_timeseries_n100.npy"
    )
    args.ts_3 = "01_fullTs_BW3_exclude_pj_IonSB_ERA5wet_spatial_correlation_stddev_timeseries_n100.npy"
    args.ts_4 = "01_fullTs_BW3_exclude_pj_IonSB_ERA5wet_ERA5dry_spatial_correlation_stddev_timeseries_n100.npy"
    args.ts_5 = "01_fullTs_BW3_exclude_pj_IonSB_ERA5wet_ERA5dry_demErr_spatial_correlation_stddev_timeseries_n100.npy"
    args.window_sizes_m = "window_sizes_m.npy"
    # args.window_sizes_m="window_sizes.npy"
    args.ts_dates = "dates_ts1.npy"

    ts_spatialcorr_raw = np.load(args.ts_1)
    ts_spatialcorr_IonSB = np.load(args.ts_2)
    ts_spatialcorr_IonSB_ERA5wet = np.load(args.ts_3)
    ts_spatialcorr_IonSB_ERA5wet_dry = np.load(args.ts_4)
    ts_spatialcorr_IonSB_ERA5wet_dry_demErr = np.load(args.ts_5)

    ts_dates = np.load(args.ts_dates, allow_pickle=True)
    window_sizes_m = np.load(args.window_sizes_m)

    # plot_window_sizes(args.png_out)
    plot_all_timesteps_window_sizes(args.png_out2)
    plot_all_timesteps_difference_window_sizes(args.png_out3)
