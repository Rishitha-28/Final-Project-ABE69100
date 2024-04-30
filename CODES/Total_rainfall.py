# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:31:04 2024

Plot the total rainfall using this file

@author: rbikkuma
"""

import matplotlib.pyplot as plt
import pandas as pd

# Specify the path to your CSV file
file_path = r"\DATA\Total_Rainfall.csv"  # Change this to the correct path

# Load the data, with a focus on parsing the datetime correctly
# Ensure the date column is named correctly as seen in the file, adjust 'DateTime' if it's named differently
data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)

# Print column names to verify if 'DateTime' is present
print("Column names in the DataFrame:", data.columns)

# Ensure the column exists before attempting to use it
if 'Date' in data.columns:  
    
    # Plotting
    plt.figure(figsize=(16,9))
    plt.bar(data['Date'], data['Rainfall'])
    plt.title('Rainfall',fontsize=25)
    plt.xlabel('Date',fontsize=20)
    plt.ylabel('Rainfall (mm)',fontsize=20)
    plt.xticks(rotation=0,fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()
else:
    print("The column 'DateTime' does not exist in the DataFrame.")
