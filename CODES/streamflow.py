# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:24:50 2024
File helps create streamflow plots for 4 nodes from 1980 t0 1982
@author: Rishitha Bikkumalla
"""
import matplotlib.pyplot as plt
import pandas as pd

filenames = [
    r"\DATA\Observed Streamflow\streamflow_node_1.csv", 
    r"\DATA\Observed Streamflow\streamflow_node_28.csv",
    r"\DATA\Observed Streamflow\streamflow_node_54.csv",
    r"\DATA\Observed Streamflow\streamflow_node_59.csv"
]

fig, axs = plt.subplots(4, 1, figsize=(10, 12))  # Creates 4 subplots
nodes=[1,28,54,59]
for i, filename in enumerate(filenames):
    # Load each CSV file
    data = pd.read_csv(filename)
    
    # Explicitly specify the date format
    data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')
    
    # Plotting the streamflow data
    axs[i].plot(data['date'], data['Discharge'])
    # axs[i].plot(data['date'], data['baseflow'])
    axs[i].set_title(f'Node {nodes[i]} Streamflow over Time',fontsize=20)
    axs[i].set_xlabel('Date',fontsize=20)
    axs[i].set_ylabel('Streamflow(cfs)',fontsize=20)

plt.tight_layout()
plt.show()



