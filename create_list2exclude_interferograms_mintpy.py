#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:32:10 2023

@author: sofia
"""

from mintpy.objects import ifgramStack
import numpy as np
import argparse
import os

# -------------------------
parser = argparse.ArgumentParser(
    description="Create list of interferograms to exclude \n after list of pairs as date2_date1",
)
parser.add_argument(
    "-l",
    "--list",
    dest="in_list",
    help="List of interferograms to exclude",
    default=None,
)
parser.add_argument(
    "-i", "--inDir", dest="inDir", help="Directory that contains \n input/ifgramStackh5"
)
args = parser.parse_args()

# ------
inDir = args.inDir
in_list = args.in_list
# --
input_file = os.path.join(inDir, "inputs/ifgramStack.h5")
dates_12_list = ifgramStack(input_file).get_date12_list(dropIfgram=False)
# ---
# Read list of intereferograms to filter
if os.path.exists(os.path.join(inDir, in_list)) is False:
    print("Input list file not found within directory")
elif in_list is None:
    print("Specify a file list of format\n\n date2_date1\ndate2_date3\n")
else:
    in_list = os.path.join(inDir, in_list)
    with open(in_list, "r") as fl:
        dates12_exclude = fl.readlines()
dates12_exclude = [pair.replace("\n", "") for pair in dates12_exclude]
dates12_exclude.sort()
# ---
dates12_exclude = np.unique(dates12_exclude)
# ---
# Get index for those dates
# ---
datesIndex_exclude = []
for date_exc in dates12_exclude:
    datesIndex_exclude.append(str(dates_12_list.index(date_exc)))

# ------
# Create a list of those interferograms
# -----
out_file = os.path.join(inDir, "dates2exclude_by_index.txt")
with open(out_file, "w") as fl:
    fl.write("{}\n".format(",".join(datesIndex_exclude)))
# ----
# Report dates/pairs to exclude
# ---
total_ifgs = len(dates_12_list)
total_excluded_ifgs = len(datesIndex_exclude)
percentage = total_excluded_ifgs * 100 / total_ifgs
datesSingle = [pair.split("_")[0] for pair in dates12_exclude]
datesSingle.extend([pair.split("_")[1] for pair in dates12_exclude])
datesSingle_unique, counts = np.unique(datesSingle, return_counts=True)

out_report = os.path.join(inDir, "report_interferogramas_excluded.txt")
with open(out_report, "w") as fl:
    fl.write("Input Dir : {} \n".format(inDir))
    fl.write("Total Ifgs : {}\n".format(total_ifgs))
    fl.write("Total Excluded Ifgs : {}\n".format(total_excluded_ifgs))
    for date, exc_date in zip(datesSingle_unique, counts):
        fl.write("Date {} : {}\n".format(date, exc_date))
