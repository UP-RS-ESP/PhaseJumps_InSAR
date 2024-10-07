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
import os,glob, time, logging
import xarray as xr
import argparse
import matplotlib.pyplot as plt
from argparse import RawTextHelpFormatter

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

#---------------------------
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

Oct-2024, Sofia Viotto (viotto1@uni-potsdam.de)
"""
#-------------------------

parser = argparse.ArgumentParser(
                    description=DESCRIPTION, epilog=EXAMPLE, formatter_class=RawTextHelpFormatter
                    )
parser.add_argument('--inDir', '-i',dest='inDir',help='Full path to the ESD folder')
parser.add_argument('--subswath', '-s',dest='subswath',
                    help='Optional. Define a sub-swath to calculate statistics',
                    nargs='+',default=None)
args = parser.parse_args()

#------------------------------------------------#


def check_input_directories(inps):
    skip=False
    # Check if the main input directory exists
    if os.path.exists(inps['inDir']) is False:
        skip=True
        #logging.info('Input Directory does not exist \n')
        logging.info('Input Directory does not exis.')
        return inps,skip
    # Create ESD_azimuth_offsets directory if not found
    elif os.path.exists(os.path.join(inps['inDir'],'ESD_azimuth_offsets')) is False:
        os.mkdir(os.path.join(inps['inDir'],'ESD_azimuth_offsets'))
        
    # Check if the ESD directory exists
    if os.path.exists(inps['ESD_dir']) is False:
        skip=True
        logging.info('ESD directory not found')
        return inps,skip
        
    else:
        # Identify sub-swaths based on the ESD folder
        ESD_pairs_folders=sorted(glob.glob(os.path.join(inps['ESD_dir'],'2*')))
        if inps['subswath']==None:
            ESD_pairs_subswath_folders=sorted(glob.glob(os.path.join(inps['ESD_dir'],'2*','IW*')))
        
            subswath_list=[os.path.basename(i) for i in ESD_pairs_subswath_folders]
            subswath_unique=np.unique(subswath_list)
        else:
            subswath_unique=inps['subswath']
        for iw in subswath_unique:
            ESD_offset_filename=sorted(glob.glob(os.path.join(inps['ESD_dir'],'2*',iw,'combined.off.vrt')))
            if len(ESD_pairs_folders)!=len(ESD_offset_filename):
                skip=True
                logging.info('Skipping. Number of pairs in the ESD folder differs from the number of combined.off.vrt files.')
                return inps,skip
            
            inps['subswath']=list(subswath_unique)
            logging.info('Sub-swath found {}'.format(subswath_unique))
            return inps,skip

def MAD(x):
    med = np.median(x)
    x   = abs(x-med)
    MAD = np.median(x)
    return MAD

def IQR(x):
    return np.nanpercentile(x,75)-np.nanpercentile(x,25)

def plot_distribution_per_burst_overlap(df_stats_medians,df_coh_points,subswath,inps):
    
    boxprops = dict(facecolor='lightblue', color='black', linewidth=0.75)
    medianprops = dict(color='red', linewidth=1)
    whiskerprops = dict(color='black', linewidth=0.75)
    capprops = dict(color='black', linewidth=0.75)
    import matplotlib.ticker as mticker
    median_max=np.nanmax(df_stats_medians.iloc[:, :-9])

    fig,axs=plt.subplots(nrows=2,figsize=(8,15/2.54))
    # First boxplot (for medians)
    axs[0].boxplot(df_stats_medians.iloc[:, :-9].values, patch_artist=True,  boxprops=boxprops, 
               medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
    axs[0].set_ylabel('Median Offset Azimuth [px]')
    axs[0].set_ylim(0,np.round(median_max,2))
    axs[0].set_title('Medians')


    # Second boxplot (for coherent points)
    axs[1].boxplot(df_coh_points.iloc[:, :-4].values, patch_artist=True,  boxprops=boxprops, 
               medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
    axs[1].set_ylabel('#Points Coh >0.85')
    axs[1].set_title('Coherent Points')
    
    # Rotate x-axis labels
    for ax in axs:
        ax.set_xlabel('Burst Overlapping Area')
        ax.tick_params(axis='x', labelrotation=90)
    axs[1].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # Set the main title and format the figure
    fig.suptitle('Statistics on Each Burst Overlap (Sub-swath {})'.format(subswath), fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(os.path.join(inps['inDir'],'ESD_azimuth_offsets/','boxplot_stats_at_burst_overlapping_area_{}.png'.format(subswath)),dpi=300)
def report_pairs(df_stats_medians,inps):
    #mads=df_stats_medians.groupby('RefDate')['MAD_px'].median()
    threshold=0.0009#np.nanpercentile(mads, 99)
    pairs=df_stats_medians[df_stats_medians['MAD_px']>=threshold].index.to_list().copy()
    out_report=os.path.join(inps['inDir'],'exclude_pairs_ESD.txt')
    with open(out_report,'w') as fl:
        fl.write('Pairs with Median Absolute Deviation MAD larger than {}\n'.format(threshold))
        fl.write("\n".join(pairs))
        
    
def plot_histograms_of_global_variables(df_stats_medians,df_coh_points,subswath,inps):
    

    fig,axs=plt.subplots(nrows=2,figsize=(8,15/2.54))
    # First boxplot (for medians)
    values=df_stats_medians['MAD_px'].values.flatten()
    n_bins=20
    axs[0].hist(values,bins=n_bins)
    axs[0].set_ylabel('Frequency (log-scale)')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('MAD per pair [px]')
    axs[0].set_title('Median Absolute Deviation of Burst Overlap')
    p75=np.nanpercentile(values,75)
    axs[0].axvline(p75,c='orange',lw=1,label='75th %ile')
    p90=np.nanpercentile(values,90)
    axs[0].axvline(p90,c='red',lw=1,label='90th %ile')
    #axs[0].text(p90, axs[0].get_ylim()[1]/3, '90th %ile', color='red', ha='center', va='center',rotation=90)

    p95=np.nanpercentile(values,95)
    axs[0].axvline(p95,c='red',lw=1,ls='--',label='95th %ile')
    #axs[0].text(p95, axs[0].get_ylim()[1]/3, '95th %ile', color='red', ha='center', va='center',rotation=90)

    accuracy_threshold=0.0009
    axs[0].axvline(accuracy_threshold,c='k',lw=1,label='Accuracy Thresh.')
    axs[0].legend()
    #axs[0].text(accuracy_threshold, axs[0].get_ylim()[1]/3, 'Accuracy Thresh.', color='red', ha='center', va='center',rotation=90)

    # Second boxplot (for coherent points)
    values=df_coh_points['TotalCohPts'].values
    axs[1].hist(values,bins=n_bins)
    axs[1].set_ylabel('Frequency (log-scale)')
    axs[0].set_yscale('log')
    axs[1].set_xlabel('Total of Coherent Points per pair')
    axs[1].set_title('Coherent Points')
    

    # Set the main title and format the figure
    fig.suptitle('Statistics on Each Burst Overlap (Sub-swath {})'.format(subswath), fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(os.path.join(inps['inDir'],'ESD_azimuth_offsets/','histograms_stats_at_burst_overlapping_area_{}.png'.format(subswath)),dpi=300)


def calculate_median_ESD_per_burst(combined_fname):
    """
    Calculates median ESD statistics for each burst overlap.

    Parameters:
    combined_fname (str): Filename of the combined offset file.

    Returns:
    Tuple containing medians, standard deviations, coherent points, and the number of burst overlaps.
    """
    # Load the combined offset, coherence, and interferogram files
    ESD_off=xr.open_dataarray(combined_fname)
    ESD_cor=xr.open_dataarray(combined_fname.replace('.off.vrt','.cor.vrt'))
    ESD_int=xr.open_dataarray(combined_fname.replace('.off.vrt','.int.vrt'))
    
    # Step 1: Mask combined offsets with coherence and interferogram thresholds
    ESD_off=ESD_off.where(ESD_cor>0.3)
    ESD_off=ESD_off.where(np.angle(ESD_int)>0)
    ESD_off=ESD_off.squeeze()
   
    # Retrieve burst overlap coordinates
    max_per_coordinates=ESD_off.max(dim='x')
    
    #Keep the coordinates were maximum values are different from zero
    coordinates=max_per_coordinates[(max_per_coordinates.notnull())].y.values

    #Group coordinates to find the y-coordinate ranges along that separates every
    #burst overlap
    coordinates_split=np.split(coordinates, np.where(np.diff(coordinates) >1)[0] + 1)
    number_brst_ovlp=len(coordinates_split)

    # Step 2: Filter pixels with coherence >= 0.85 (ESD threshold)
    ESD_off=ESD_off.where(ESD_cor>0.849)
    
    # Calculate median, std, and number of coherent points per burst overlap
    medians=[]
    std=[]
    coh_points=[]
    for group in coordinates_split:
        medians.append(np.nanmedian(ESD_off.sel(y=group).data))
        std.append(np.nanstd(ESD_off.sel(y=group).data))
        coh_points.append(np.count_nonzero(~np.isnan(ESD_off.sel(y=group).data)))
        
    return medians,std,coh_points,number_brst_ovlp


def calculate_stats_by_subwath(inps,subswath):
    """
    Calculates and saves ESD statistics for each sub-swath.

    Parameters:
        inps (dict): Input directory paths.
        subswath (str): Sub-swath identifier.
    """
    # Find files in the ESD directory
    ESD_offset_filename=sorted(glob.glob(os.path.join(inps['ESD_dir'],'2*',subswath,'combined.off.vrt')))
    n_pairs=len(ESD_offset_filename)
    medians,std,coh_points,n_brst_ovlp=[],[],[],[]

    for fname in ESD_offset_filename:
        median_brst_ovlp,std_brst_ovlp,coh_point_brst_ovlp,number_brst_ovlp=calculate_median_ESD_per_burst(combined_fname=fname)
        medians.extend(median_brst_ovlp)
        std.extend(std_brst_ovlp)
        coh_points.extend(coh_point_brst_ovlp)
        n_brst_ovlp.append(number_brst_ovlp)
        
    #Check number of burst overlapping areas
    if len(set(n_brst_ovlp)) == 1:
        logging.info(f'Number of burst overlapping areas found: {n_brst_ovlp[0]} ({subswath})')
    else:
        logging.info('Error found during calculations.')
        
    
    # Reshape lists 
    std=np.asarray(std).reshape(n_pairs,n_brst_ovlp[0])
    medians=np.asarray(medians).reshape(n_pairs,n_brst_ovlp[0])
    coh_points=np.asarray(coh_points).reshape(n_pairs,n_brst_ovlp[0])
    pairs=[os.path.basename(fname.split('/'+subswath)[0]) for fname in ESD_offset_filename]
    
    #Coordinates are always read from the first burst overlapping area to the last one
    burst_overlap=['BstOvlp' + str(i) for i in range(1,n_brst_ovlp[0]+1)]
    
    #--------------------------------------#
    #Prepare dataframe and save them
    #Dataframe of median azimuth offset per burst overlapping areas
    df_stats_medians=pd.DataFrame(medians,columns=burst_overlap,index=pairs)
    df_stats_medians=df_stats_medians.add_prefix('MedianAzOff_')
    df_stats_medians=df_stats_medians.add_suffix('_px')
    #Dataframe
    df_stats_std=pd.DataFrame(std,columns=burst_overlap,index=pairs)
    df_stats_std=df_stats_std.add_prefix('StdAzOff_')
    df_stats_std=df_stats_std.add_suffix('_px')
    
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
    df_stats_medians['MAD_px']=mads
    df_stats_medians['IQR_px']=iqrs
    
    
    #Add other parameters
    df_stats_medians['RefDate']= [pd.to_datetime(i.split('_')[0],format='%Y%m%d') for i in df_stats_medians.index.tolist()]
    df_stats_medians['RefDate_month']= df_stats_medians['RefDate'].dt.month

    df_stats_medians['SecDate']= [pd.to_datetime(i.split('_')[1],format='%Y%m%d') for i in df_stats_medians.index.tolist()]
    df_stats_medians['SecDate_month']= df_stats_medians['SecDate'].dt.month

    df_stats_medians['RefDate_year']= df_stats_medians['RefDate'].dt.year
    df_stats_medians['SecDate_year']= df_stats_medians['SecDate'].dt.year

    df_stats_medians['Bt_days']=(df_stats_medians['SecDate']-df_stats_medians['RefDate']).dt.days
    
    #----------------------------------------------#
    #Prepare the dataframe of coherent points per burst overlap
    df_coh_points['TotalCohPts']=df_coh_points.sum(axis=1)
    df_coh_points['RefDate']=df_stats_medians['RefDate'].copy()
    df_coh_points['SecDate']=df_stats_medians['SecDate'].copy()
    df_coh_points['Bt_days']=df_stats_medians['Bt_days'].copy()
    
    #--------------------------------------------------#
    #Save dataframes  
    logging.info('Saving dataframes')
    df_stats_medians.to_csv(os.path.join(inps['inDir'],'ESD_azimuth_offsets/ESD_azimuth_offset_medians_pairs_{}.csv'.format(subswath)),
                            float_format='%.15f')
    df_stats_std.to_csv(os.path.join(inps['inDir'],'ESD_azimuth_offsets/ESD_azimuth_offset_std_pairs_{}.csv'.format(subswath)),
                        float_format='%.15f')
    df_coh_points.to_csv(os.path.join(inps['inDir'],'ESD_azimuth_offsets/ESD_azimuth_offset_coh_points_pairs_{}.csv'.format(subswath)),
                         float_format='%.15f')
    
    #--------------------------------------------------#
    #Save summaries from dataframes
    df_stats_medians_describe=df_stats_medians.iloc[:,:-7].describe()
    df_stats_medians_describe.to_csv(os.path.join(inps['inDir'],'ESD_azimuth_offsets/summary_ESD_azimuth_offset_medians_pairs_{}.csv'.format(subswath)),
                                     float_format='%.15f')
    df_coh_points_describe=df_coh_points.iloc[:,:-7].describe()
    df_stats_medians_describe.to_csv(os.path.join(inps['inDir'],'ESD_azimuth_offsets/summary_ESD_azimuth_offset_coh_points_pairs_{}.csv'.format(subswath)),
                                     float_format='%.15f')
    #-----------------------------------------------------#
    #Plot
    logging.info('Plotting figures')
    plot_distribution_per_burst_overlap(df_stats_medians,df_coh_points,subswath,inps)
    plot_histograms_of_global_variables(df_stats_medians,df_coh_points,subswath,inps)
    
    #----------------------------------------------------#
    #Report
    logging.info('Reporting pairs with large MAD of Azimuth Offset')
    report_pairs(df_stats_medians,inps)
    
def run():
    
    inps={'inDir':os.path.dirname(os.path.abspath(args.inDir)),
          'ESD_dir':os.path.abspath(args.inDir),
          'subswath':args.subswath}
    logging.info('Checking input parameters')
    inps,skip=check_input_directories(inps)
    if skip==False:
        logging.info('Retrieving burst overlap statistics at the sub-swath level')
        for subswath in inps['subswath']:
            calculate_stats_by_subwath(inps,subswath)
        
run()
