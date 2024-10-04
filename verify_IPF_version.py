#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 17:51:37 2023

@author: Sofia Viotto
"""

# ------------------------------------#
# The script performs an assessment of IPF versions from input scenes
# ------------------------------------#
# This requieres two environments
# ISCE2
# MintPy
# ------------------------------------#
import os
import sys
import glob
import zipfile
import argparse
import datetime
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd


# Create a class to store IPF versions from input file
class sentinelSLC(object):
    """Class defined according to script
    s1_select_ion.py from Cunren Liang, 22-MAR-2018
    """

    def __init__(self, safe_file, orbit_file=None):
        self.safe_file = safe_file
        self.orbit_file = orbit_file

    def get_datetime(self):
        # Date format from input zip files
        datefmt = "%Y%m%dT%H%M%S"
        safe = os.path.basename(self.safe_file)
        # Define parameters of the input scene according to names
        fields = safe.split("_")
        # Get plataform
        self.platform = fields[0]
        # Get starting date to compose information about input scenes
        self.start_date_time = datetime.datetime.strptime(fields[5], datefmt)

    def get_param(self):

        # Open file and read the scene metadata
        fl = zipfile.ZipFile(self.safe_file, "r")
        # Read manifest
        manifest = [item for item in fl.namelist() if ".SAFE/manifest.safe" in item][0]
        # Retrieve information
        xmlFl = fl.read(manifest)
        root = ET.fromstring(xmlFl)
        elem = root.find(
            './/metadataObject[@ID="processing"]'
        )  # Encuentra la version de procesamiento

        # setup namespace
        nsp = "{http://www.esa.int/safe/sentinel-1.0}"
        rdict = elem.find(".//xmlData/" + nsp + "processing/" + nsp + "facility").attrib
        self.proc_site = rdict["site"] + ", " + rdict["country"]

        rdict = elem.find(
            ".//xmlData/" + nsp + "processing/" + nsp + "facility/" + nsp + "software"
        ).attrib
        self.proc_version = rdict["version"]


class orbit(object):
    """A class for obit files"""

    def __init__(self, orbit_file):
        self.orbit_file = orbit_file

    def get_quality(self):
        # Read file
        fl = self.orbit_file
        with open(fl, "r") as foo:
            lines = foo.readlines()
            foo.close()
        quality = [i for i in lines if "<Quality>" in i]
        quality = quality[0].split("</")[0].split(">")[-1]
        self.quality = quality


def get_information_safe(dir_slc):
    """
    this function check for input scenes and retrive information about IPF versions

    """
    # sort by starting time
    zips = sorted(
        glob.glob(os.path.join(dir_slc, "S1*_IW_SLC_*.zip")),
        key=lambda x: x.split("_")[-5],
        reverse=False,
    )
    nzips = len(zips)

    safe_list = []
    for i in range(nzips):
        safeObj = sentinelSLC(zips[i])
        safeObj.get_datetime()
        safeObj.get_param()
        safe_list.append(
            {"file": safeObj.safe_file, "IPF Version": safeObj.proc_version}
        )
    return safe_list


def get_information_orbit(dir_orbit):
    """
    This function search for orbits files and read quality parameters of the orbits
    to check that orbits are optimal
    """

    orbits = sorted(glob.glob(os.path.join(dir_orbit, "*EOF")))

    orbit_files_out = []
    for orbit_i in orbits:
        orbitObj = orbit(orbit_i)
        orbitObj.get_quality()

        orbit_files_out.append(
            {
                "file": orbit_i,
                "quality": orbitObj.quality,
            }
        )

    return orbit_files_out


def reportOrbits(orbit_list):
    """
    Check qualiy of input orbits and report it if the quality is not NOMINAL
    https://sentinels.copernicus.eu/documents/247904/4599719/Copernicus-POD-Product-Handbook.pdf
    """
    list_orbits_check = []
    for dct_i in orbit_list:
        if dct_i["quality"] != "NOMINAL":
            list_orbits_check.append(dct_i)

    if list_orbits_check != []:
        out_orbits_report = "report_orbits.txt"

        print("*" * 50)
        print("\n")
        print("Orbits with degraded quality found \n")

        print("Reporting orbits in file: {} \n".format(out_orbits_report))
        with open(out_orbits_report, "w") as fl:
            for item in list_orbits_check:
                fl.write(item["file"] + "\n")
    else:

        print("Orbits with degraded quality were not found")


def reportSafe(safe_list):
    """
    this function report scenes with IPF versions below 2.6
    """
    list_scenes_out = []
    for safe in safe_list:
        if float(safe["IPF Version"]) < 2.6:
            list_scenes_out.append(safe["file"])
    if list_scenes_out != []:
        report_out = "scenes_IPF_vesions.txt"
        print("*" * 50)
        print("\n")
        print("scenes with IPF version < 2.6 \n")
        print("Reporting to file : {}\n".format(report_out))
        with open(report_out, "w") as fl:
            for item in list_scenes_out:
                fl.write("%s\n" % item)
                fl.close()
        print(
            "A file to exclude dates by IPF versions was created \n exclude_by_IPF.txt"
        )
        foo = [os.path.basename(i) for i in list_scenes_out]
        dates = [i.split("_")[5].split("T")[0] for i in foo]
        dates = list(np.unique(dates))
        string_out = ",".join(dates)

        with open("exclude_by_IPF.txt", "w") as fl:
            fl.write(string_out)
            fl.close()
    else:
        print("There is not images with IPF versions below 2.6")


# First, Given an input directory check IPF versions of input scenes
# Scenes with IPF versiones < 2.6 are discarded


def cmdLineParse():
    """
    Command line parser.
    """

    parser = argparse.ArgumentParser(
        description="Report bad orbits and scenes with IPF versions < 2.6"
    )
    parser.add_argument(
        "-dirSlc",
        dest="dirSlc",
        type=str,
        required=True,
        help='directory with the "S1*_IW_SLC_*.zip" files',
    )
    parser.add_argument(
        "-dirOrb",
        dest="dirOrb",
        type=str,
        required=True,
        help="directory with orbits files *EOF",
    )

    if len(sys.argv) <= 1:
        print("")
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == "__main__":

    inps = cmdLineParse()

    # Report scenes with IPF versions < 2.6
    safe_list = get_information_safe(inps.dirSlc)
    reportSafe(safe_list)

    # Report quality of orbits
    orbits_list = get_information_orbit(inps.dirOrb)
    reportOrbits(orbits_list)
