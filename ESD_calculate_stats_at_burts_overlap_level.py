#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:37:41 2024

@author: Sofia Viotto
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os,glob
import xarray as xr
import argparse

parser = argparse.ArgumentParser(
                    description='Calculate stats from ESD files',
                    )
parser.add_argument('--inDir', '-i',dest='inDir',help='Parent directory that contains the ESD folder')
args = parser.parse_args()

#------------------------------------------------#


def check_input_directories(inps):
    skip=False
    if os.path.exists(inps['inDir']) is False:
        skip=True
        print('Input Directory does not exist \n')
    if os.path.exists(os.path.join(inps['inDir'],'spreadsheet_csv')) is False:
        os.mkdir(os.path.join(inps['inDir'],'spreadsheet_csv'))
    if os.path.exists(inps['ESD_dir']) is False:
        skip=True
        print('ESD directory not found \n')
        
    else:
        #Define the sub-swath based on the folder ESD
        ESD_pairs_folders=sorted(glob.glob(os.path.join(inps['ESD_dir'],'2*','IW*')))
        
        subswath_list=[os.path.basename(i) for i in ESD_pairs_folders]
        subswath_unique=np.unique(subswath_list)
        for iw in subswath_unique:
            ESD_offset_filename=sorted(glob.glob(os.path.join(inps['ESD_dir'],'2*',iw,'combined.off.vrt')))
            if len(ESD_pairs_folders)!=len(ESD_offset_filename):
                skip=True
                print('Skiping. Number of pairs inside the ESD parent folder different to Number of combined.off.vrt files \n ')
            else:
                inps['sub_swath']=list(subswath_unique)
    return inps,skip

def MAD(x):
    med = np.median(x)
    x   = abs(x-med)
    MAD = np.median(x)
    return MAD
def IQR(x):
    return np.nanpercentile(x,75)-np.nanpercentile(x,25)

    
def calculate_median_ESD_per_burst(combined_fname):
    #For each pair, open 
    #combined.off: combined azimuth offset file after ESD, 
    #combined.cor: combined coherence at the burst overlapping areas after ESD
    #combined.int: combined double difference interferogram used to retrieve the combined azimuth offset
    
    ESD_off=xr.open_dataarray(combined_fname)
    ESD_cor=xr.open_dataarray(combined_fname.replace('.off.vrt','.cor.vrt'))
    ESD_int=xr.open_dataarray(combined_fname.replace('.off.vrt','.int.vrt'))
    
    #------
    #STEP 1:
    #combined.off: every burst overlap is separated by rows filled with zero data values
    #Use a lower threshold of coherence as we want to detect the coordinates of every burst 
    #overlapping area
    #Then combine coherence and double difference interferograms to define those coordinates
   
    ESD_off=ESD_off.where(ESD_cor>0.3)
    ESD_off=ESD_off.where(np.angle(ESD_int)>0)
    ESD_off=ESD_off.squeeze()
   
    #Retrieve coordinates of every burst overlapping area by calculating the maximum per
    #row. 
    #If the value is zero== the row is filled with zero values
    #If the value is not zero== the row belongs to a burst overlapping area
    max_per_coordinates=ESD_off.max(dim='x')
    
    #Keep the coordinates were maximum values are different from zero
    coordinates=max_per_coordinates[(max_per_coordinates.notnull())].y.values

    #Group coordinates to find the range along the y dimension that separates every
    #burst overlap
    coordinates_split=np.split(coordinates, np.where(np.diff(coordinates) >1)[0] + 1)
    number_brst_ovlp=len(coordinates_split)

    #--------------------------------
    #STEP 2:
    # keep only the pixels with coherence => 0.85,
    #as the threshold used during calculations with ESD
    ESD_off=ESD_off.where(ESD_cor>0.849)
    
    # Retrieve median, std and number of coherence points (i.e. values not masked out) at burst overlap level
    medians=[]
    std=[]
    coh_points=[]
    for group in coordinates_split:
        medians.append(np.nanmedian(ESD_off.sel(y=group).data))
        std.append(np.nanstd(ESD_off.sel(y=group).data))
        coh_points.append(np.count_nonzero(~np.isnan(ESD_off.sel(y=group).data)))
    return medians,std,coh_points,number_brst_ovlp

def calculate_stats_by_subwath(inps,sub_swath):
    #Find files
    ESD_offset_filename=sorted(glob.glob(os.path.join(inps['ESD_dir'],'2*',sub_swath,'combined.off.vrt')))
    n_pairs=len(ESD_offset_filename)
    medians,std,coh_points,n_brst_ovlp=[],[],[],[]

    for fname in ESD_offset_filename:
        median_brst_ovlp,std_brst_ovlp,coh_point_brst_ovlp,number_brst_ovlp=calculate_median_ESD_per_burst(combined_fname=fname)
        medians.extend(median_brst_ovlp)
        std.extend(std_brst_ovlp)
        coh_points.extend(coh_point_brst_ovlp)
        n_brst_ovlp.append(number_brst_ovlp)
        #Check all values number of burst overlap are the same among different pairs
        
    if len(set(n_brst_ovlp)) == 1:
        print('Number of burst overlaping areas found {} at sub-swath {}\n'.format(n_brst_ovlp[0],sub_swath))
    else:
        print('Error found during calculations \n')

    #Reshape lists to arrays
    std=np.asarray(std).reshape(n_pairs,n_brst_ovlp[0])
    medians=np.asarray(medians).reshape(n_pairs,n_brst_ovlp[0])
    coh_points=np.asarray(coh_points).reshape(n_pairs,n_brst_ovlp[0])
    pairs=[os.path.basename(fname.split('/'+sub_swath)[0]) for fname in ESD_offset_filename]
    
    #Coordinates are always read from the first burst overlapping area to the last one
    burst_overlap=['BstOvlp' + str(i) for i in range(1,n_brst_ovlp[0]+1)]
    #--------------------------------------#
    #Prepare dataframe and save them

    df_stats_medians=pd.DataFrame(medians,columns=burst_overlap,index=pairs)
    df_stats_medians=df_stats_medians.add_prefix('MedianAzOff_')
    df_stats_medians=df_stats_medians.add_suffix('[px]')
    
    df_stats_std=pd.DataFrame(std,columns=burst_overlap,index=pairs)
    df_stats_std=df_stats_std.add_prefix('StdAzOff_')
    df_stats_std=df_stats_std.add_suffix('[px]')
    
    df_coh_points=pd.DataFrame(coh_points,columns=burst_overlap,index=pairs)
    df_coh_points=df_coh_points.add_prefix('CohPts_')

    #Transpose to calculate MADs per pair
    df_stats_medians_T=df_stats_medians.T.copy()

    iqrs=[]
    mads=[]
    for i in df_stats_medians_T.columns.tolist():
        mads.append(MAD(df_stats_medians_T[i].values))
        iqrs.append(IQR(df_stats_medians_T[i].values))
    
    #Add MADs,IQRs, Range of medians across burst overlapping areas to dataframe
    
    df_stats_medians['MAD[px]']=mads
    df_stats_medians['IQR[px]']=iqrs
    df_stats_medians['Range(Max-Min)']=df_stats_medians.max(axis=1)-df_stats_medians.min(axis=1)
    
    #Add other parameters
    df_stats_medians['Ref_Date']= [pd.to_datetime(i.split('_')[0],format='%Y%m%d') for i in df_stats_medians.index.tolist()]
    df_stats_medians['Ref_Date_Month']= df_stats_medians['Ref_Date'].dt.month

    df_stats_medians['Sec_Date']= [pd.to_datetime(i.split('_')[1],format='%Y%m%d') for i in df_stats_medians.index.tolist()]
    df_stats_medians['Sec_Date_Month']= df_stats_medians['Sec_Date'].dt.month

    df_stats_medians['Ref_Date_Year']= df_stats_medians['Ref_Date'].dt.year
    df_stats_medians['Sec_Date_Year']= df_stats_medians['Sec_Date'].dt.year

    df_stats_medians['Bt(days)']=(df_stats_medians['Sec_Date']-df_stats_medians['Ref_Date']).dt.days
    
    #----------------------------------------------#
    #Prepare the dataframe of coherent points per burst overlap
    df_coh_points['Sum_Coh_points']=df_coh_points.sum(axis=1)
    df_coh_points['Ref_Date']=df_stats_medians['Ref_Date'].copy()
    df_coh_points['Sec_Date']=df_stats_medians['Sec_Date'].copy()
    df_coh_points['Bt(days)']=df_stats_medians['Bt(days)'].copy()
    
    #--------------------------------------------------#
    #Save dataframes  

    df_stats_medians.to_csv(os.path.join(inps['inDir'],'spreadsheet_csv/ESD_azimuth_offset_medians_pairs_{}.csv'.format(sub_swath)))
    df_stats_std.to_csv(os.path.join(inps['inDir'],'spreadsheet_csv/ESD_azimuth_offset_std_pairs_{}.csv'.format(sub_swath)))
    df_coh_points.to_csv(os.path.join(inps['inDir'],'spreadsheet_csv/ESD_azimuth_offset_nonan_pairs_{}.csv'.format(sub_swath)))


def run():
    
    inps={'inDir':args.inDir,
          'ESD_dir':os.path.join(args.inDir,'ESD')}
    
    inps,skip=check_input_directories(inps)
    if skip==False:
        for sub_swath in inps['sub_swath']:
            calculate_stats_by_subwath(inps,sub_swath)
        
run()