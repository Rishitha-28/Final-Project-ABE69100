# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:49:52 2024
This file helps create the statistics and metrics of the project
@author: Rishitha Bikkumalla
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.ticker as ticker

#Functions

def ReadData(fileName):
    # Define column names
    colNames = ['Date','Baseflow','Discharge']

    # Open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,
                         parse_dates=['Date'], infer_datetime_format=True)
    DataDF = DataDF.set_index('Date')

    # Quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()

    return(DataDF, MissingValues)
def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
        a 1 year time series of streamflow, after filtering out NoData
        values.  Tqmean is the fraction of time that daily streamflow
        exceeds mean streamflow for each year. Tqmean is based on the
        duration rather than the volume of streamflow. The routine returns
        the Tqmean value for the given data array."""
    mean_flow = Qvalues.mean()
    exceedances = Qvalues[Qvalues > mean_flow].count()
    Tqmean = exceedances / len(Qvalues)
    return Tqmean
    

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
        (R-B Index) of an array of values, typically a 1 year time
        series of streamflow, after filtering out the NoData values.
        The index is calculated by dividing the sum of the absolute
        values of day-to-day changes in daily discharge volumes
        (pathlength) by total discharge volumes for each year. The
        routine returns the RBindex value for the given data array."""
    # Ensure there are enough data points to calculate the index
    if Qvalues.count() < 2:
        return np.nan  # Need at least two data points to compute differences

    daily_changes = Qvalues.diff().abs().sum()
    total_volume = Qvalues.sum()

    return daily_changes / total_volume if total_volume != 0 else np.nan

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
        values, typically a 1 year time series of streamflow, after 
        filtering out the NoData values. The index is calculated by 
        computing a 7-day moving average for the annual dataset, and 
        picking the lowest average flow in any 7-day period during
        that year.  The routine returns the 7Q (7-day low flow) value
        for the given data array."""
    seven_day_min = Qvalues.rolling(window=7).mean().min()
    return seven_day_min

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
        than 3 times the annual median flow. The index is calculated by 
        computing the median flow from the given dataset (or using the value
        provided) and then counting the number of days with flow greater than 
        3 times that value.   The routine returns the count of events greater 
        than 3 times the median annual flow value for the given data array."""
    median_flow = Qvalues.median()
    count_exceedances = Qvalues[Qvalues > 3 * median_flow].count()
    return count_exceedances

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    # Resample to annual (water year) and calculate metrics
    WYDataDF = DataDF.resample("AS-OCT").apply(lambda x: {
        
        
        "Mean Flow": x['Discharge'].mean(),
        "Peak Flow": x['Discharge'].max(),
        "Median Flow": x['Discharge'].median(),
        "Coeff Variation": (x['Discharge'].std() / x['Discharge'].mean()) * 100,
        "Skew": stats.skew(x['Discharge'], nan_policy='omit'),
        "TQmean": CalcTqmean(x['Discharge']),
        "R-B Index": CalcRBindex(x['Discharge']),
        "7Q": Calc7Q(x['Discharge']),
        "3xMedian": CalcExceed3TimesMedian(x['Discharge'])
    }).apply(pd.Series)
    WYDataDF.columns = [ "Mean Flow", "Peak Flow", "Median Flow", "Coeff Var", "Skew", 
                        "Tqmean", "R-B Index", "7Q", "3xMedian"]
    return WYDataDF
    

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""
    MoDataDF = DataDF.resample("MS").apply(lambda x: {
        
         
        "Mean Flow": x['Discharge'].mean(),
        "Coeff Variation": (x['Discharge'].std() / x['Discharge'].mean()) * 100,
        "TQmean": CalcTqmean(x['Discharge']),
        "R-B Index": CalcRBindex(x['Discharge'])
    }).apply(pd.Series)
   
    MoDataDF.columns = [ "Mean Flow", "Coeff Var", "Tqmean", "R-B Index"]
    return MoDataDF


def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    AnnualAverages = WYDataDF.mean()
    return AnnualAverages

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    MonthlyAverages = MoDataDF.groupby(MoDataDF.index.month).mean()
    return MonthlyAverages

#Files with sreamflow data 
fileName = { 'Node1':r"Final_Project\DATA\Observed Streamflow\streamflow_node_1.csv",
             'Node28': r"Final_Project\DATA\Observed Streamflow\streamflow_node_28.csv",
             'Node59' : r"Final_Project\DATA\Observed Streamflow\streamflow_node_59.csv"}
# define blank dictionaries (these will use the same keys as fileName)
DataDF = {}
MissingValues = {}
WYDataDF = {}
MoDataDF = {}
AnnualAverages = {}
MonthlyAverages = {}
Annual_DF=pd.DataFrame()
Monthly_DF=pd.DataFrame()
for file in fileName.keys():
    # print(file)
    
    # print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
    
    DataDF[file], MissingValues[file] = ReadData(fileName[file])
    # print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
    print(DataDF[file])
    print('-----')
    # clip to consistent period
    
    # calculate descriptive statistics for each water year
    WYDataDF[file] = GetAnnualStatistics(DataDF[file])
    
    # calcualte the annual average for each stistic or metric
    AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
    
    print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])
   
    # calculate descriptive statistics for each month
    MoDataDF[file] = GetMonthlyStatistics(DataDF[file])
 
    # calculate the annual averages for each statistics on a monthly basis
    MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
    
    print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])
    
    temp_df=WYDataDF[file]
    temp_df["Station"]=file
    Annual_DF = pd.concat([Annual_DF, temp_df])
    
    temp1_df=MoDataDF[file]
    temp1_df["Station"]=file
    Monthly_DF = pd.concat([Monthly_DF, temp1_df])

#Daily FLow Figure
plt.figure(figsize=(16, 9), dpi=(1280/16))
for text_file in fileName.keys():
    # Read the data from the text file
    DataDF[file], MissingValues[file] = ReadData(fileName[file])

    # Plot the daily flow
    plt.plot(DataDF[text_file].index, DataDF[text_file]['Discharge'], label=text_file)
plt.title('Daily Flow', fontsize=25)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Discharge (cfs)', fontsize=20)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig('Daily_FLow.png')
plt.show()


#Metrics Image
fig, axs = plt.subplots(3, 1, figsize=(16, 9), dpi=(1280/16))
metrics = ['Coeff Var', 
           'Tqmean', 
           'R-B Index']
metrics_names = {'Coeff Var':'Coefficient of Variation', 
                      'Tqmean': 'T-Qmean', 
                      'R-B Index':'Richards-Baker Flashiness Index'}
lines_labels = [] 
for i, metric in enumerate(metrics):
    # Create a bar plot for the current metric
    annual_data = Annual_DF.pivot(columns='Station', values=metric)
    print('======')
    print(annual_data)
    print('======')
    annual_data.plot(kind='bar', ax=axs[i])
                     # , rot=A2E20)
   
    xticks = axs[i].get_xticks()
    # axs[i].set_xticks(xticks[::3])

    labels=[1979, 1980,1981,1982]
    axs[i].set_xticklabels(labels,fontsize=20,rotation=0)
    axs[i].set_title(f'Annual {metrics_names[metric]}',fontsize=25)
    axs[i].set_xlabel('Year',fontsize=20)
    axs[i].set_ylabel(metric,fontsize=20)
    axs[i].legend(title='Station',loc='best',fontsize=20)
    # axs[i].tight_layout()
    

plt.tight_layout()
plt.savefig('Annual_Parameters.png')
plt.show()


#Monthly flow
   
# Ensure the 'Date' column is in datetime format and set as index
Monthly_DF.index = pd.to_datetime(Monthly_DF.index)

# Group by both month and station, then calculate the mean for each group
monthly_avg_flow_per_station = Monthly_DF.groupby([Monthly_DF.index.month, 'Station'])['Mean Flow'].mean()

# Unstack to create a DataFrame where each station is a column and each row represents a month
monthly_avg_flow_per_station = monthly_avg_flow_per_station.unstack(level='Station')
print('---')
print(monthly_avg_flow_per_station.index)
# Plot the Average Annual Monthly Flow for both the rivers
# Ensure size and quality of image
fig, ax = plt.subplots(figsize=(16, 9), dpi=96)  # Adjust dpi to match your presentation requirements

# Plot each station's data with different colors, if necessary
for station in fileName.keys():
    print(station)
    ax.plot(monthly_avg_flow_per_station.index, monthly_avg_flow_per_station[station], marker='o',  label=station)

# Setting custom x-axis labels to be month names
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# ax.set_xticklabels([ 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Adjust for proper alignment
ax.set_xticklabels([ 'Jan','Feb',  'Apr',  'Jun',  'Aug',  'Oct',  'Dec'])  # Adjust for proper alignment
ax.set_title('Average Annual Monthly Flow', fontsize=25)
ax.set_xlabel('Month', fontsize=20)
ax.set_ylabel('Average Flow (cfs)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.legend( fontsize=20)
ax.grid(True)

plt.tight_layout(pad=1.0)  # Adjust padding if necessary
plt.savefig('Average_annual_monthly_flow.png', bbox_inches='tight')  # Ensure the entire plot is saved
plt.show()