# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:42:37 2024

@author: Rishitha Bikkumalla

This code can be used to make plots of Predicted streamflow and NSE data:
    1)Create Linear Interpolation Map
    2)NSE PIE chart
    3)Correlation charts
    
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import pearsonr

NSE_df = pd.read_csv(r"C:\Users\rbikkuma\OneDrive - purdue.edu\Courses\EnvInformatics\Final_Project\DATA\Predicted\Parameters_samples_1000.csv")

#Create Linear Interpolation Map
# Create pivot table
pivot_df = NSE_df.pivot(index="Tc_factor", columns="S_factor", values="NSE")

# Define smaller grid size
tc = np.linspace(NSE_df['Tc_factor'].min(), NSE_df['Tc_factor'].max(), 100)
s = np.linspace(NSE_df['S_factor'].min(), NSE_df['S_factor'].max(), 100)

# Create meshgrid
TC, S = np.meshgrid(tc, s)
threshold=-1000
# Interpolate values onto the grid points
points = NSE_df[['Tc_factor', 'S_factor']].values
values = NSE_df['NSE'].values
NSE_grid = griddata(points, values, (TC, S), method='linear')


NSE_grid = np.where(NSE_grid < threshold, np.nan, NSE_grid)
max_nse_value = pivot_df.max().max()
max_tc, max_s = pivot_df.stack().idxmax()

# Plot the heatmap
plt.figure(figsize=(8, 6))
plt.pcolormesh(S, TC, NSE_grid, shading='gouraud', cmap='viridis')
plt.colorbar(label='NSE')
plt.scatter( max_s, max_tc, color='red', s=100, marker='o', edgecolors='black', zorder=2)
plt.scatter(NSE_df['S_factor'], NSE_df['Tc_factor'], color='black', s=5, alpha=0.2)
plt.title(f'Linear Interpolated NSE Heatmap')
plt.xlabel('Storage factor')
plt.ylabel('Tc factor')
plt.show()

print(f"Maximum NSE value: {max_nse_value:.3f} at Tc factor = {max_tc}, S factor = {max_s}")


#NSE PIE CHART
# Calculate percentages of positive and negative NSE values
positive_nse = len([x for x in NSE_df["NSE"] if x > 0])
negative_nse = len(NSE_df["NSE"]) - positive_nse
percent_positive = (positive_nse / len(NSE_df["NSE"])) * 100
percent_negative = (negative_nse / len(NSE_df["NSE"])) * 100
print(positive_nse,negative_nse)
# Create pie chart
sizes = [percent_positive, percent_negative]
colors = ['#48737A', '#7A2E20']
explode = (0.1, 0)  # explode 1st slice
plt.figure(figsize=(9, 6))
plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

# Add legend
plt.legend(labels=['Positive NSE', 'Negative NSE'], loc="best")

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Percentage of Positive vs Negative NSE Values')

plt.show()

#Correlation charts
data = NSE_df
nse_tc_corr, nse_tc_pvalue = pearsonr(data['NSE'], data['Tc_factor'])
print(f"Correlation between NSE and Tc_factor: {nse_tc_corr:.3f} (p-value: {nse_tc_pvalue:.3e})")

# Correlation between NSE and S_factor
nse_s_corr, nse_s_pvalue = pearsonr(data['NSE'], data['S_factor'])
print(f"Correlation between NSE and S_factor: {nse_s_corr:.3f} (p-value: {nse_s_pvalue:.3e})")

plt.figure(figsize=(16,9), dpi=(1280/16))

plt.scatter(data['Tc_factor'], data['NSE'])
plt.xlabel('Tc',fontsize=20)
plt.ylabel('NSE',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title(f'Correlation between NSE and Tc',fontsize=25)
plt.show()

plt.figure(figsize=(16,9), dpi=(1280/16))
plt.scatter(data['S_factor'], data['NSE'])
plt.xlabel('S',fontsize=20)
plt.ylabel('NSE',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title(f'Correlation between NSE and S',fontsize=25)
plt.show()