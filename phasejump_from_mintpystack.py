#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#------------------------#
#Author Sofia Viotto
# Year: 2024


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os,glob
from mintpy.objects import ifgramStack
from mintpy.utils import readfile
#import seaborn as sns
import argparse
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt
import numba as nb
import datetime
from datetime import date
import copy
from mintpy.utils import utils as ut

#---------------------------------------#
#Plotting styles
plt.rcParams['font.family'] = 'Sans'
plt.rcParams['font.style'] = 'normal'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

#-----------------------------------#
synopsis = 'Extract azimuth gradient from unwrapped phase in radar coordinates and find phase jumps'
EXAMPLE='''
    python phasejump_from_mintpystack.py --in_dir /path/mintpy  --plotIndividual --burst-number 9 \n 
    python phasejump_from_mintpystack.py --in_dir /path/mintpy --pair 20160524_20160711 --burst-number 9  --percent-min 85 \n 

    
    References
    1) Wang et al., 2017 "Improving burst aligment in tops interferometry with BESD",
    10.1109/LGRS.2017.2767575
    2) Zhong et al., 2014 "A Quality-Guided and Local Minimum Discontinuity Based 
    Phase Unwrapping Algorithm for InSAR InSAS Interograms", 10.1109/LGRS.2013.2252880
    ******************************************************************
    
'''

parser = argparse.ArgumentParser(
                    description=synopsis,epilog=synopsis,usage=EXAMPLE
                    )
parser.add_argument('--inDir', '-i',dest='in_dir',help='Input folder containning input/ifgramStack.h5 in RADAR COORDINATES',
default=os.getcwd())
parser.add_argument('--plotIndividual',dest='plotIndividual',help='plot individual interferograms next to gradient along the azimuth direction \n',
                    action='store_true')
parser.add_argument('--pair','-p', default=None,help='Input pair',dest='pair')
parser.add_argument('--burst-number',dest='bursts',type=int,help='Specify the number of bursts expected within the dataset, which helps to determine the probability of a an area jumping, of being a phase jump')
parser.add_argument('--percentage-min',default=85, help='Minimun percentage of pixels jumping along a row to be defined as phase jump. It is not recomendable to change the thershold, as the rows may be note reliable')
args= parser.parse_args()
inps = args.__dict__

#------------------------------------#
def check_inputs(inps):
    #----
    #Check folder
    #----
    #print('Checking input parameters\n')
    skip=False
    if os.path.exists(inps['stack_fname']) is False:
        print('inputs/ifgramStack.h5 not found\n')
        skip=True
        return skip   
    #-----
    #Check datastet is not geocoded
    if os.path.exists(inps['stack_fname']):
        atr=readfile.read_attribute(inps['stack_fname'])
        if 'Y_FIRST' in atr.keys():
            print('The script is adapted for Not Geocoded datasets, as the result may be inpredectible')
            skip=True
            

    return skip

#-------Plot fuctions

def PlotData_UnwGradient(inps):
    #Arrays
    ds_unw = inps['ds_unw']
    ds_abs_grad = inps['ds_abs_grad']
    severity_alongX_proportion = inps['severity_alongX_proportion']
    percentage_min=inps['percentage_min']
    #Figure name
    name = inps['name']
    date12 = inps['date12']
    orbit = inps['orbit']
    if 'fig_size' not in inps.keys(): fig_size=(10,7)
    
    title = 'Pair %s (Mask Coherence)' % date12
    
    # Plot   
    fig, axs = plt.subplots(ncols=3, figsize=fig_size, sharex=False, 
                            sharey=True, gridspec_kw={'width_ratios': [3, 3, 1]})
    fig.subplots_adjust(top=0.8)
    # Title for the entire figure
    fig.suptitle(title, fontsize=11, fontweight='bold')

    #    Subplot 1: Unwrapped Phase
    axs[0].set_title(r'Unwrap Phase $\varphi$', fontsize=11)
    unwPlot = axs[0].imshow(ds_unw, cmap='RdBu', interpolation='nearest',
                        vmin=np.nanpercentile(ds_unw, 10),
                        vmax=np.nanpercentile(ds_unw, 90))
    fig.colorbar(unwPlot, label=r'$\varphi$ [Rad]', ax=axs[0], pad=0.03, shrink=0.6, aspect=30, 
                             orientation='vertical')
    axs[0].set_ylabel('Y (Radar)')
    axs[0].set_xlabel('X (Radar)')
    axs[1].set_xlabel('X (Radar)')

    # Subplot 2:
    axs[1].set_title(r'Absolute ($\varphi_{(i+1,j)}-\varphi_{(i,j)}$)', fontsize=11)
    vmax = np.nanpercentile(ds_abs_grad, 98)
    vmin = np.nanpercentile(ds_abs_grad, 2)
    gradPlot = axs[1].imshow(ds_abs_grad, cmap='viridis', interpolation='none',
                         vmin=vmin, vmax=vmax)
    fig.colorbar(gradPlot, ax=axs[1], label=r'$|\delta\varphi_{az}|$ [mm]', pad=0.03, shrink=0.6, aspect=30, orientation='vertical')

    # Subplot 3: Jumps Proportion
    axs[2].set_title(r'$\Sigma$ $Jumps_{(i)}$',fontsize=11)
    axs[2].plot(severity_alongX_proportion, np.arange(0, severity_alongX_proportion.shape[0], 1),
            lw=0.5, c='black')
    axs[2].set_xlabel('Jumps [%Pixels]')
    axs[2].set_xticks([0, 50, 100])
    axs[2].set_xticklabels([0, 50, 100])
    axs[2].axvline(x=percentage_min, c='r', lw=0.5, zorder=0)

    # Axis inversion based on orbit
    if orbit.lower().startswith('a'):
        axs[0].invert_yaxis()

    elif orbit.lower().startswith('d'):
        axs[0].invert_xaxis()
        axs[1].invert_xaxis()
        # Set aspect ratio to keep all plots the same height
    for ax in axs[:2]:
        ax.set_aspect('auto')
        axs[2].set_aspect(aspect='auto', adjustable='box')
    # Adjust layout for better spacing
    fig.subplots_adjust(wspace=0.3)
    
    fig.savefig(name,dpi=300)

    
    plt.clf()
    plt.close()        
    

def analyse_PhaseJump_by_date(inps):
    """
    Analyzes and identifies phase phase jumps in overlapping areas of bursts.
    phase jumps in the overlapping area are distinguished by the fact that
    they are located at approximately regular intervals.

    It reads severity files and identifies phase phase jumps at regular intervals.
    Coordinates found are reported.

    Parameters:
    -----------
    inps : dict
        A dictionary containing the following keys:
        - 'date12List' : list
            List of date pairs. Format 'YYYYMMDD_yyyymmdd'
    
        - 'percentage_min' : float
            Minimum height percentage to consider a peak of detected pixels as a phase jump, default is 90% of the pixels along the row.

        - 'in_dir' : str
            Directory path where output reports will be saved.
        - 'in_dir_arr_1d' : str
            Directory path containing 1D severity files.
        - 'in_dir_arr_2d' : str
            Directory path containing 2D severity files.
        - 'outDir_figure' : str
            Directory path where output figures will be saved.
        - 'fname_absolute_azimuth_gradient' : str
            Filename of the 2D azimuth gradient file.
        - 'bursts' : int
            Number of bursts in the dataset.
  
    """ 
    

    #Input dates
    date12List=inps['date12List']
    burst_number=inps['bursts']
    percentage_min=inps['percentage_min']
    #---------------#
    #Inputs files
    fname_abs_az_grad=inps['fname_absolute_azimuth_gradient']
    fname_severity_1D=inps['fname_severity_1D']
    
    # Load datasets
    phase_jumps=xr.open_dataarray(fname_severity_1D)
    ds_abs_grad=xr.open_dataarray(fname_abs_az_grad)
    
    #Output reports
    outReport_phase_date12=os.path.join(inps['in_dir'],'phase_jumps_per_date.txt')
    outReport_summary=os.path.join(inps['in_dir'],'summary_phase_jumps_y_coordinates.txt')
    outReport_exclude_list_interferograms=os.path.join(inps['in_dir'],'exclude_listdate12_interferograms_by_phase_jump.txt')
    outReport_exclude_list_dates=os.path.join(inps['in_dir'],'exclude_dates_by_phase_jumps.txt')
    
    
    #----------------
    #STAGE 1:
    #Determine regular coordinates
    #---------------
    Y_regular_spacing=ds_abs_grad.Y.max().item()//burst_number
    print('\n*Burst Ovlp Areas must be located at ~ %s pixels \n'%Y_regular_spacing)

    #------------------
    #STAGE 2:
    #store coordinates with phase jump at every severity file
    #----------------
    
    phase_jumps=phase_jumps.where(phase_jumps>percentage_min)
    
    #Drop pairs with no phase_jumps
    phase_jumps=phase_jumps.dropna(dim='pair',how='all')
    #Drop coordinates without phase_jumps
    phase_jumps=phase_jumps.dropna(dim='Y',how='all')
    
    date12phase_jumps=list(phase_jumps.pair.values)

    phase_jumps_allDates=[]
    
    #-----Rough position
    print('\n*Estimating position of Burst Overlap based on sample of size %s unwrap phase'%phase_jumps.pair.shape[0])
    for idx,date12 in enumerate(date12phase_jumps):
        
        # Find phase_jumps with height between a minimun percentage and 100% of the row
        phase_jumps_pair=phase_jumps.isel(pair=idx).copy()
        y_coord_phase_jumps_pair=phase_jumps_pair.dropna(dim='Y').Y.values
        
        # Remove coordinates that do not follow the pattern of being regular coordinates
        # as expected
        y_coord_diff2regular_coordinates=np.abs(y_coord_phase_jumps_pair-(y_coord_phase_jumps_pair//Y_regular_spacing)*Y_regular_spacing)
        
        #Given filter stage and multilooking, plus differences on the overlapping areas
        #allow a maximum difference of the phase jump of 20 pixels between the regular position
        #and the position where the phase jump is found
        y_coord_phase_jumps_pair=y_coord_phase_jumps_pair[y_coord_diff2regular_coordinates<20]
        #Remove coordinates at the border
        #If those were detected
        y_coord_phase_jumps_pair=y_coord_phase_jumps_pair[y_coord_phase_jumps_pair<((burst_number*Y_regular_spacing)-20)]

        phase_jumps_allDates.extend(list(y_coord_phase_jumps_pair))
    
    
    #----------------
    #STAGE 3:
    #--Refinement of positions
    #----------------
    print('\n*Refining coordinates based on frequency of phase jumps')
    pj,fr=np.unique(phase_jumps_allDates,return_counts=True)
    fr,pj=fr[pj>(Y_regular_spacing-5)],pj[pj>(Y_regular_spacing-5)]
    #Group closer coordinates 
    y_groups=np.split(pj, np.where(np.diff(pj) > 20)[0] + 1)
    fr_groups=np.split(fr, np.where(np.diff(pj) > 20)[0] + 1)
    y_temp=[]
     
    for gp,fr_gp in zip(y_groups,fr_groups):
        if gp.shape[0]>1:
                try:
                    y_temp.append(gp[fr_gp==np.max(fr_gp)].item())
                except:
                    y_temp.append(gp[0])
        else:
                y_temp.append(gp[0])
                
                
    print('\nFrequent coordinates %s \n'%y_temp)
    phase_jumps=phase_jumps.sel(Y=np.asarray(y_temp))
    phase_jumps=phase_jumps.dropna(dim='pair',how='all')
    #----------------------------------------#
    #
    #----Redefiny phase jumps by assesing the magnitude of the same
    ds_abs_grad_phase_jump=ds_abs_grad.sel(Y=np.asarray(y_temp)).copy() 
    
    median_pery_perpair_phase_jump=ds_abs_grad_phase_jump.median(dim=('X'))
    #Add the median of those coordinates to the global stats
    values=[]
    for pair in median_pery_perpair_phase_jump.pair.values:
        arr=median_pery_perpair_phase_jump.sel(pair=pair).copy()
        arr=arr.sel(Y=np.asarray(y_temp))
        arr=arr.data.flatten()
        arr=np.round(arr,2)
        arr=list(arr)
        
        arr=[str(i) for i in arr]
        
        values.append(','+",".join(arr)+'\n')
        del arr
    
    #Open stats and add the new stats:
    outFile_stats=os.path.join(inps['in_dir'],'stats_absolute_gradient.txt')
    with open(outFile_stats,'r') as fl:
        lines=fl.readlines()
        fl.close()
    y_temp=['AbsGrad_Median_Y-'+str(int(i))+'[mm]' for i in y_temp]
    lines[10]=lines[10].replace('\n',","+",".join(y_temp)+"\n")
    #Add the statis
    subset_lines=copy.deepcopy(lines[11:])
    subset_lines=[i.replace('\n',j) for i,j in zip(subset_lines,values)]
    lines[11:]=subset_lines
    with open(outFile_stats,'w') as fl:
        for line in lines:
            fl.write(line)
        fl.close()
    
    
    
    #--------------------#
    #STAGE 4
    #-------------------#´

    
    median_threshold=np.nanpercentile(median_pery_perpair_phase_jump.data,95)
        
    median_pery_perpair=median_pery_perpair_phase_jump.where(median_pery_perpair_phase_jump>=median_threshold)
    median_pery_perpair=median_pery_perpair.dropna(dim='pair',how='all')
    date12phase_jumps_final=list(median_pery_perpair.pair.values)
    
    print('\n>> Redefining significative unwrap phase jumps.\n*Total of pairs found with significative phase jumps: %s\n'%len(date12phase_jumps_final))
    print('\n Phase jump is significative if the median jump of the line is >%s mm\n'%np.round(median_threshold,2))
    
    # #--------------Beging Report
    phase_jumps_allDates=[]
    UnwrapPhase2Report=[]
    y_coord_phase_jumps_pair=[]
    
    for idx,date12 in enumerate(date12phase_jumps_final):
        
        # Find phase_jumps with height between a minimun percentage and 100% of the row
        ds_abs_grad_pair=median_pery_perpair.sel(pair=date12).copy()
        y_coord_phase_jumps_pair=list(ds_abs_grad_pair.dropna(dim='Y').Y.values)
       
        
        if len(y_coord_phase_jumps_pair)>0:
            #Calculate the median phase jump
            median_abs_grad_pair=np.nanmedian(ds_abs_grad_pair.data)
                
            #Save outputs and reports
            phase_jumps_allDates.extend(y_coord_phase_jumps_pair)
            y_coord_phase_jumps_pair=[str(i) for i in y_coord_phase_jumps_pair]
            spacing='\t'*((burst_number//2-len(y_coord_phase_jumps_pair)//2))
            UnwrapPhase2Report.extend([date12+'\t'+str(len(y_coord_phase_jumps_pair))+'\t\t'+','.join(y_coord_phase_jumps_pair)+spacing+str(np.round(median_abs_grad_pair,2))])
        else:
            #Remove date12 if there is not date to report
            date12phase_jumps_final=[i for i in date12phase_jumps_final if i!=date12]
            continue
        del ds_abs_grad_pair

          
    #----------------------------------------------#
    #Report phase jumps
    #----------------------------------------------#    
    if len(UnwrapPhase2Report) > 0:
        
            
            header = '# Pairs found with systematic phase jumps \n'
            header += '# N_Disc: Number of phase jumps \n'
            header += '# Y_coord: Coordinates along the azimuth direction\n\n'
            
            #-Report each pair, number of phase jumps and coordinates  
            with open(outReport_phase_date12,'w') as filehandle:
                    filehandle.write(header)
                    filehandle.write('#\t DATE12 \tN_Disc\t\tY_Coord\t\t\tMedian_Phase_Jump[mm]\n')
                    for item in UnwrapPhase2Report:
                        filehandle.write('%s\n'%item)
    
    #Provide a summary of the coordinates found
    phase_jumps_,counts_=np.unique(phase_jumps_allDates,return_counts=True)
    with open(outReport_summary,'w') as filehandle:
        filehandle.write('#Y_coord\tCounts\n')
        for pos,cnt in zip(phase_jumps_, counts_):
            filehandle.write('%s\t%d\n'%(pos,cnt))
    # #------------------        
    # #Create a list of interferograms

    with open(outReport_exclude_list_interferograms,'w') as fl:
        fl.write(",".join(date12phase_jumps_final))
        
    # #--------------------------    
    #Create a list to exclude dates from further processing based on how many interferograms
    #With phase jumps were found per date
    dates=[pair.split('_')[0] for pair in date12List]
    dates.extend([pair.split('_')[1] for pair in date12List])
    dates,fr_interferograms=np.unique(dates,return_counts=True)
    
    dateswithphase_jumps=[i.split('_')[0] for i in date12phase_jumps_final]
    dateswithphase_jumps.extend(i.split('_')[1] for i in date12phase_jumps_final)
    dateswithphase_jumps,fr_ifgs_phase_jumps=np.unique(dateswithphase_jumps,return_counts=True)
    
    proportion=np.asarray([n_ifgs_phase_jump*100/fr_interferograms[dates==date_i] for date_i,n_ifgs_phase_jump in zip(dateswithphase_jumps,fr_ifgs_phase_jumps) ]).flatten()
    
    dd=list(dateswithphase_jumps[proportion>49])
    #print(dd)
    with open(outReport_exclude_list_dates,'w') as fl:
        fl.write(",".join(dd))
        
def save_global_stats2txt(inps):
    
    refList=[datetime.datetime.strptime(pair.split('_')[0],'%Y%m%d') for pair in inps['date12List']]
    secList=[datetime.datetime.strptime(pair.split('_')[1],'%Y%m%d') for pair in inps['date12List']]
    Bt=[(sec-ref).days for ref,sec in zip(refList,secList)]
    coherence_stats=np.asarray(inps['coherence_stats'])#[:,:-1]
    abs_grad_stats=np.asarray(inps['abs_grad_stats'])
    size=1+coherence_stats.shape[1]+abs_grad_stats.shape[1]
    
    array=np.empty((len(refList),size),dtype=float)
    array[:,0]=Bt
    array[:,1:coherence_stats.shape[1]+1]=coherence_stats
    array[:,coherence_stats.shape[1]+1:]=abs_grad_stats
    
    
    #Save Txt file with stats from Azimuth Gradient
    outFile=os.path.join(inps['in_dir'],'stats_absolute_gradient.txt')
    
    header='#No Data Values (Zero Values) were skipped from all calculations \n'
    header+='# Coherence statistics were retrieved from all No Nans pixels .\n'
    header+='# Pixels with coherence <0.75 were masked out in the calculation of the absolute Azimuth Gradient statistics and corresponding severity.\n'
    header+='# The number of masked-out pixels varies from pair to pair.\n'
    header+='##Column Names/Prefix: \n'
    header+='#Btemp: Temporal Baseline \n'
    header+='#Coh: Coherence\n'
    header+='#Abs_Grad: Absolute Gradient Along the Azimuth Direction \n'
    header+='#Pxs: Number of No-Nan Pixels \n\n'


    stats_name=['AbsGrad_Median[mm]','AbsGrad_Mean[mm]','AbsGrad_Std[mm]']       
    #Prepare name of columns
    colCoh=copy.deepcopy(stats_name)
    colCoh=[i.replace('AbsGrad','Coh').replace('[mm]','') for i in colCoh ]
    colCoh=",".join(colCoh)
    colabs_grad=stats_name
    colabs_grad=",".join(colabs_grad)
    columns='DATE12,Btemp[days],'+colCoh+','+colabs_grad+'\n'

    with open(outFile, 'w') as filehandle:
        filehandle.write(header)
        filehandle.write(columns)
        for line,pair in zip(array,inps['date12List']):
            line=list(np.round(line,2).astype(str))
            line=pair+','+",".join(line)+'\n'
            filehandle.write(line)
        
        
#-------------------------#
@nb.njit(parallel=True)
def diff_along_azimuth(ds_unw):
    #ds_grad_az in shape (time,rows-Y,cols-X)
    ds_grad_az = np.empty(ds_unw.shape, dtype=np.float32)
    for n in nb.prange(ds_unw.shape[0]):
        array = np.concatenate((np.zeros((1, ds_unw.shape[2]), dtype=np.float32), 
                                ds_unw[n,:,:].astype(np.float32))).astype(np.float32)
        
        ds_grad_az[n,:,:] = np.diff(array.T).T.astype(np.float32)
    return ds_grad_az



@nb.njit(parallel=True)
def stats_ds_unw(ds_unw):
    
    stats = np.empty((ds_unw.shape[0],3), dtype=np.float32)
    
    for n in nb.prange(ds_unw.shape[0]):
        array=ds_unw[n,:,:]
        stats[n,0]=np.round(np.nanmedian(array),2)
        stats[n,1]=np.round(np.nanmean(array),2)
        stats[n,2]=np.round(np.nanstd(array),2)    
        #stats[n,3]=np.count_nonzero(~np.isnan(array))
        
    
    return stats


def readData2VerticalGradient(inps):
    
    #----------------Define parameters
    stack_fname=inps['stack_fname']
    pbar=inps['pbar']
    outDir_arr_2d=inps['outDir_arr_2d']
    outDir_arr_1d=inps['outDir_arr_1d']
    
    #----------------------------------
    #Define outputs name
    if inps['pair']==None:
            
            #Define output names
            fname_counts_nonan=os.path.join(outDir_arr_1d,'Counts_NoNan.nc')
            fname_severity_1D=os.path.join(outDir_arr_1d,'severity_absolute_azimuth_gradient_along_range.nc')
            fname_severity_2D=os.path.join(outDir_arr_2d,'severity_absolute_gradient_azimuth.nc')
            fname_abs_grad_2D=os.path.join(outDir_arr_2d,'absolute_azimuth_gradient_2D_mm.nc')
    
    else:
            #Define output names
            date12=inps['pair']
            fname_counts_nonan=os.path.join(outDir_arr_1d,date12+'_Count_NoNan.nc')
            fname_severity_1D=os.path.join(outDir_arr_1d,date12+'_severity_absolute_azimuth_gradient_along_range.nc')
            fname_severity_2D=os.path.join(outDir_arr_2d,date12+'_severity_absolute_gradient_azimuth.nc')
            fname_abs_grad_2D=os.path.join(outDir_arr_2d,date12+'_absolute_azimuth_gradient_2D_mm.nc')
            
    
    #-------------Begining Main Operations-------------#
  
    
    #-------------
    #Calculate severity of the phase jump using a  threshold
    #threshold=coefficient*np.pi
    #-------------
    # A phase jump will be defenitvely happend if the threshold=2 * pi for the unwrap Phase,
    # As used in Eq 2, page 216
    # Minor phase jumps also occur for smaller values of pi
    #.---------------------
    #For Sentinel 1: lambda 0.05546576 m = 5.54 cm
    #A coefficient of is 0.50 equals to  a phase jump of (pi/2), which is 0.635 cm
    # A coefficient of 1 equals to a phase jump of (1 *pi), which is is 1.385 cm
    
   
    
    #---------------
    #Read unwrap Phase
    #---------------
    atr=readfile.read_attribute(stack_fname)
    inps['atr']=atr
    
    #Load the whole dataset if pair not specified
    if inps['pair']==None:
        ds_unw=readfile.read(stack_fname, datasetName='unwrapPhase')[0]
    else:
        ds_unw=readfile.read(stack_fname, datasetName='unwrapPhase-%s'%inps['pair'])[0]
    
        
    
    #Zero is no data in the wrapped phase, and it must be as well
    #in the unw phase
    ds_unw[ds_unw==0]=np.nan
    #print(ds_unw.shape)
    #--------------------
    #Calculate Azimuth Gradient
    #-----------------
    #Calculate the diff along the azimuth direction to later compute
    #v(m,n)  from the equation of Zhong et al., 2014:
    #v m,n = Int((φ m,n − φ m−1,n )/2π) (Eq 2, page 216)
    #ds_grad_az= (φ m,n − φ m−1,n )
    #axis 0 is equal to Y direction=azimuth direction
    #ds_grad_az=np.diff(ds_unw, axis=0,prepend=0)

    #Absolute Azimuth Gradient  and related outputs and parameters
    #will be called with subfix  abs_grad
    if inps['pair']==None:
        
        #Foor loop with prepend  
        ds_abs_grad=np.abs(diff_along_azimuth(ds_unw))

    else:
        ds_abs_grad=np.abs(np.diff(ds_unw, axis=0,prepend=0))
    
    pbar.update(1)
    #Express in milimiters
    phase2range =float(atr['WAVELENGTH']) / (4.*np.pi)
    ds_abs_grad*=phase2range
    ds_abs_grad*=1000
    #Clean memory 
    del ds_unw
    
    
    #---------------------
    #Read coherence & Use it as mask
    #--------------------
    if inps['pair']==None:
        #Foor loop with prepend  
        ds_coh=readfile.read(stack_fname, datasetName='coherence')[0]
    else:
        ds_coh=readfile.read(stack_fname, datasetName='coherence-%s'%inps['pair'])[0]
    #Zero is No Data
    ds_abs_grad[ds_coh<0.75]=np.nan

    
    #-----------------
    #Stats from datasets
    #----------------  
  
    #Other parameters for stats
    if inps['pair']==None:
        stats_ds_abs_grad=stats_ds_unw(ds_abs_grad)    
        ds_coh[ds_coh==0]=np.nan
        stats_coh=stats_ds_unw(ds_coh)
       
        #Clean memory
        del ds_coh
        inps['coherence_stats']=stats_coh
        inps['abs_grad_stats']=stats_ds_abs_grad
        save_global_stats2txt(inps)
        del stats_coh,stats_ds_abs_grad
    else:
        print('\n >> Skipping global statistics for a unique dataset.. \n')
    pbar.update(1)

    

    #-----------------------------------------
    #Calculate the severity
    #-----------------------------------------
    #Calculate the severity in 2D, as the division
    # between the Azimuth Gradient and and the threshold,
    #then round the values to zero decimals.
    #By rounding, 0 = phase jumps below and far away from ratio gradient to threshold
    #1=phase jumps close or above 1 from the 
    #
    #Obtain the median for every time step
    if inps['pair']==None:
            coefficient=np.nanpercentile(ds_abs_grad,50,axis=(1,2)) 
            #Transpose to (dim_0,dim_1,time) and transpose back 
            severity_2d=np.round(np.divide(ds_abs_grad.T ,coefficient),0).T
            del coefficient
            #Set values ds_abs_grad /(coefficient*np.pi) > 1 , also to one
            severity_2d[severity_2d>1]=1
            axis_number=2
            #Prepare to save it
            ds_abs_grad=xr.DataArray(ds_abs_grad,dims=('pair','Y','X'),coords={
                                    'pair':inps['date12List'],
                                    'Y':np.arange(0,ds_abs_grad.shape[1],1),
                                    'X':np.arange(0,ds_abs_grad.shape[2],1)})
    else:
            coefficient=np.nanpercentile(ds_abs_grad,50) 
            severity_2d=np.round(np.divide(ds_abs_grad,coefficient),0)
            del coefficient
            #Set values ds_abs_grad /(coefficient*np.pi) > 1 , also to one
            severity_2d[severity_2d>1]=1
            axis_number=1
            #Prepare to save it 
            ds_abs_grad=xr.DataArray(ds_abs_grad,dims=('Y','X'),coords={
                                    'Y':np.arange(0,ds_abs_grad.shape[1],1),
                                    'X':np.arange(0,ds_abs_grad.shape[2],1)})
    
    
    
    print('\n >> Saving Gradient ... \n')
    ds_abs_grad=ds_abs_grad.rename('absolute_azimuth_gradient_mm')
    ds_abs_grad.to_netcdf(fname_abs_grad_2D,encoding={'absolute_azimuth_gradient_mm':{'dtype': 'float32','zlib':True, 'complevel':7}})
    del ds_abs_grad
    pbar.update(1)

    #Count along the X axis how many times 1 is found
    severity_alongX=np.nansum(severity_2d,axis=axis_number)

   
    #Exprese severity as the proportion of counts to
    #NoNan pixels along X direction, which is the range direction
    NoNan_counts=np.count_nonzero(~np.isnan(severity_2d),axis=axis_number)
    NoNan_counts=NoNan_counts.astype(int)
    #--------------------#
    #Remove coordinates that are smaller than a certain percentage of NoNanPixels
    NoNan_counts_p10=np.nanpercentile(NoNan_counts,10)
    
    severity_alongX[NoNan_counts<NoNan_counts_p10]=np.nan
    np.seterr(invalid='ignore')
    severity_alongX_proportion=np.divide(severity_alongX,NoNan_counts)*100
    #Set first two rows and the two last rows to nan to avoid phase_jumps at the border of the dataset
    severity_alongX_proportion=np.round(severity_alongX_proportion,1)
    del severity_alongX,NoNan_counts_p10
        
    
        
    # #----------------Begining Save Outputs ------------------------------#           
    # #--------------------------#
    # #Save
    # #--------------------------#
    # severity_2d=severity_2d.astype(int)
    if inps['pair']==None:
            
            print('\n >> Saving Other Relevant files... \n')
            
            #Convert to xarray, save and clean memory
            NoNan_counts=xr.DataArray(NoNan_counts,dims=('pair','Y'),
                                      coords={'pair':inps['date12List'],
                                        'Y':np.arange(0,NoNan_counts.shape[1],1)})
            NoNan_counts=NoNan_counts.rename('NoNanCounts') 
            NoNan_counts.to_netcdf(fname_counts_nonan,encoding={'NoNanCounts':{'dtype': 'int16','zlib':True, 'complevel':7}})
            del NoNan_counts
            
            severity_alongX_proportion=xr.DataArray(severity_alongX_proportion,
                                                    dims=('pair','Y'),
                                                    coords={'pair':inps['date12List'],
                                                            'Y':np.arange(0,
                                                    severity_alongX_proportion.shape[1],1)})
           
            severity_alongX_proportion=severity_alongX_proportion.rename('severity_accumulated_along_range')
            severity_alongX_proportion.to_netcdf(fname_severity_1D,encoding={'severity_accumulated_along_range':{'dtype': 'int16','zlib':True, 'complevel':7,'_FillValue':-999}})
            del severity_alongX_proportion   

            severity_2d=xr.DataArray(severity_2d,dims=('pair','Y','X'),
                                  coords={'pair':inps['date12List'],
                                      'Y':np.arange(0,severity_2d.shape[1],1)})
            severity_2d=severity_2d.rename('severity_along_azimuth')
            severity_2d.to_netcdf(fname_severity_2D,encoding={'severity_along_azimuth':{'dtype': 'int16','zlib':True, 'complevel':7,'_FillValue':-999}})
            del severity_2d
            

            # ds_abs_grad=xr.DataArray(ds_abs_grad,dims=('pair','Y','X'),coords={
            #                         'pair':inps['date12List'],
            #                         'Y':np.arange(0,ds_abs_grad.shape[1],1),
            #                         'X':np.arange(0,ds_abs_grad.shape[2],1)})
            # ds_abs_grad=ds_abs_grad.rename('absolute_azimuth_gradient_mm')
            # ds_abs_grad.to_netcdf(fname_abs_grad_2D,encoding={'absolute_azimuth_gradient_mm':{'dtype': 'float32','zlib':True, 'complevel':7}})
            # del ds_abs_grad
            
    else:

            #Prepare arrays
            NoNan_counts=xr.DataArray(NoNan_counts,dims=('Y'),coords={'Y':np.arange(0,NoNan_counts.shape[0],1)})
            NoNan_counts=NoNan_counts.assign_coords({'pair':date12})
            NoNan_counts=NoNan_counts.expand_dims('pair')
            NoNan_counts=NoNan_counts.rename('NoNanCounts') 
            NoNan_counts.to_netcdf(fname_counts_nonan,encoding={'NoNanCounts':{'dtype': 'int16','zlib':True, 'complevel':7}})
            del NoNan_counts
            
            severity_alongX_proportion=xr.DataArray(severity_alongX_proportion,dims=('Y'),coords={'Y':np.arange(0,severity_alongX_proportion.shape[0],1)})
            severity_alongX_proportion=severity_alongX_proportion.assign_coords({'pair':date12})
            severity_alongX_proportion=severity_alongX_proportion.expand_dims('pair')
            severity_alongX_proportion=severity_alongX_proportion.rename('severity_accumulated_along_range')
            severity_alongX_proportion.to_netcdf(fname_severity_1D,encoding={'severity_accumulated_along_range':{'dtype': 'int16','zlib':True, 'complevel':7,'_FillValue':-999}})
            del severity_alongX_proportion            

            severity_2d=xr.DataArray(severity_2d,dims=('Y','X'),coords={'Y':np.arange(0,severity_2d.shape[0],1),
                                                                        'X':np.arange(0,severity_2d.shape[1],1)})
            severity_2d=severity_2d.assign_coords({'pair':date12})
            severity_2d=severity_2d.expand_dims('pair')
            severity_2d=severity_2d.rename('severity_along_azimuth')    
            severity_2d.to_netcdf(fname_severity_2D,encoding={'severity_along_azimuth':{'dtype': 'int16','zlib':True, 'complevel':7,'_FillValue':-999}})
            del severity_2d
            
            # ds_abs_grad=xr.DataArray(ds_abs_grad,dims=('Y','X'),coords={
            #                             'Y':np.arange(0,ds_abs_grad.shape[0],1),
            #                             'X':np.arange(0,ds_abs_grad.shape[1],1)})
            # ds_abs_grad=ds_abs_grad.assign_coords({'pair':date12})
            # ds_abs_grad=ds_abs_grad.expand_dims('pair')
            # ds_abs_grad=ds_abs_grad.rename('absolute_azimuth_gradient_mm')
            # ds_abs_grad.to_netcdf(fname_abs_grad_2D,encoding={'absolute_azimuth_gradient_mm':{'dtype': 'float32','zlib':True, 'complevel':7}})
            # del ds_abs_grad
            
        
    # # #---------------End Main Operations ---------------------------------#

    inps['fname_absolute_azimuth_gradient']=fname_abs_grad_2D
    inps['fname_severity_1D']=fname_severity_1D
    
    return inps
   

def initialisation_parameters(inps):
    
    inps['in_dir']=os.path.abspath(inps['in_dir'])
    inps['stack_fname']=os.path.join(inps['in_dir'],'inputs/ifgramStack.h5')
    inps['outDir_arr']=os.path.join(inps['in_dir'],'quality_assessment_unwrapPhase')
    inps['outDir_arr_2d']=os.path.join(inps['outDir_arr'],'2D')
    inps['outDir_arr_1d']=os.path.join(inps['outDir_arr'],'1D')
    inps['outDir_figure']=os.path.abspath(os.path.join(inps['in_dir'],'figure_phase_jump')) 
    if inps['percentage_min']==None: inps['percentage_min']=90
    if 'plotIndividual' not in inps.keys():
        inps['plotIndividual']=False
    
    return inps



def run(inps):
    inps=initialisation_parameters(inps)

    skip=check_inputs(inps)
    if skip == True:
        print('Error. Check input parameters\n')
    else:
        atr=readfile.read_attribute(inps['stack_fname'])
        #Ensure out dir exists
        if os.path.exists(inps['outDir_arr']) is False:
            os.makedirs(inps['outDir_arr'])
            os.makedirs(inps['outDir_arr_2d'])
            os.makedirs(inps['outDir_arr_1d'])
        if os.path.exists(inps['outDir_figure']) is False:
            os.makedirs(inps['outDir_figure'])
            
        #-------------
        #Add input parameters necessary to run
        #the script
        
        inps['orbit']=atr['ORBIT_DIRECTION']

        if inps['pair'] ==None:
            #Select input dates
            date12List=ifgramStack(inps['stack_fname']).get_date12_list(dropIfgram=False)
            inps['date12List']=date12List
            print('\n >> Total of unwrap Phase pairs found : %s \n'%len(date11List))

        #--------Run Calculate gradient
        with tqdm(total=4, desc="Calculating Azimuth Gradient", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            if inps['pair'] ==None:
                #Select input dates
                date12List=ifgramStack(inps['stack_fname']).get_date12_list(dropIfgram=False)
                inps['date12List']=date12List
                #Separate input dates in reference and secondarys
                ref=[date12.split('_')[0] for date12 in date12List]
                sec=[date12.split('_')[1] for date12 in date12List]
                dates,countPerDate=np.unique((ref+sec),return_counts=True)
                inps['dates']=list(dates)
                inps['N_ifgs_date']=list(countPerDate)
                #print('\n >> Total of unwrap Phase pairs found : %s \n'%len(date11List))
            
                inps['pbar']=pbar
                files=glob.glob(os.path.join(inps['outDir_arr_2d'],'*.nc'))
                
               
                if len(files)>0:
                    print('\n>> Output directory not empty. Skiping calculations \n')
                    inps['fname_absolute_azimuth_gradient']=os.path.join(inps['outDir_arr_2d'],
                        'absolute_azimuth_gradient_2D_mm.nc')
                    inps['fname_severity_1D']=os.path.join(inps['outDir_arr_1d'],
                                                           'severity_absolute_azimuth_gradient_along_range.nc')
                    inps['fname_severity_1D']=os.path.join(inps['outDir_arr_1d'],'severity_absolute_azimuth_gradient_along_range.nc')
                    pbar.update(3)
                    print('\n>>Updating stats..')
                    ds_abs_grad=xr.open_dataarray(inps['fname_absolute_azimuth_gradient'])
                    ds_abs_grad=ds_abs_grad.data
                    stats_ds_abs_grad=stats_ds_unw(ds_abs_grad)  
                    del ds_abs_grad
                    
                    ds_coh=readfile.read(inps['stack_fname'],datasetName='coherence')[0]
                    ds_coh[ds_coh==0]=np.nan
                    stats_coh=stats_ds_unw(ds_coh)
                    del ds_coh
                    inps['coherence_stats']=stats_coh
                    inps['abs_grad_stats']=stats_ds_abs_grad
                    
                    save_global_stats2txt(inps)
                    del stats_coh,stats_ds_abs_grad
                    pbar.update(1)
                    
                else:
                    inps=readData2VerticalGradient(inps)
                    pbar.update(1)
            else:
                inps['pbar']=pbar
                inps=readData2VerticalGradient(inps)  
                pbar.update(1)
        if inps['pair'] ==None:
            #-------------------------
            # Find the severity files per date as reference/secondary
            #-------------------------
            print('\n>> Analizing dicontinuities per date ..\n')
            analyse_PhaseJump_by_date(inps)
        #--------------------
        #Plot unwrapPhase next to gradient and severity of discontinuity
        #--------------------
        
        if inps['plotIndividual']==True:
            print('\n >> Plotting individual phase jumps ...\n')
            if inps['pair']==None:
                ds_abs_grad=xr.open_dataarray(inps['fname_absolute_azimuth_gradient'])
                severity_alongX_proportion=xr.open_dataarray(inps['fname_severity_1D'])
                #-
                ds_abs_grad=ds_abs_grad.squeeze()
                ds_abs_grad=ds_abs_grad.to_numpy()
                #-
                severity_alongX_proportion=severity_alongX_proportion.squeeze()
                severity_alongX_proportion=severity_alongX_proportion.to_numpy()
                #-
                for n in range(0,len(inps['date12List'])):
                    
                    date12=inps['date12List'][n]
                    inps['date12']=date12
                    name=os.path.join(inps['outDir_figure'],'unwrapPhase_absolute_vertical_gradient_%s.png'%date12)
                    da_unw=readfile.read(inps['stack_fname'], datasetName='unwrapPhase-%s'%date12)[0]

                    inps.update({'name':name,
                          'ds_unw':da_unw,
                          'ds_abs_grad':ds_abs_grad[n,:,:],
                          'severity_alongX_proportion':severity_alongX_proportion[n,:],
                         
                            })
            
                    PlotData_UnwGradient(inps)
            else:
                date12=inps['pair']
                inps['date12']=date12
                name=os.path.join(inps['outDir_figure'],'unwrapPhase_absolute_vertical_gradient_%s.png'%date12)
                da_unw=readfile.read(inps['stack_fname'], datasetName='unwrapPhase-%s'%date12)[0]
                ds_abs_grad=xr.open_dataarray(inps['fname_absolute_azimuth_gradient'])
                inps.update({'name':name,
                          'ds_unw':da_unw,
                          'ds_abs_grad':ds_abs_grad,
                        
                          'severity_alongX_proportion':severity_alongX_proportion,
                         
                            })
            
                PlotData_UnwGradient(inps)
            del da_unw
            #Clean inps to continue processing
            del inps['ds_unw']
            del inps['ds_abs_grad']
            del inps['severity_alongX_proportion']        

            
        

initialisation_parameters(inps)
run(inps)
print('Done!\n')

