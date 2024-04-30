# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:53:57 2024

@author: rbikkuma

This code creates random samples from a uniform distribution and plots them

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

num_samples=1000

#UNIFORM
param_values_placeholder  = np.random.uniform(0, 3, size=(num_samples ,2))  # Placeholder values

# Creating a DataFrame
df = pd.DataFrame(param_values_placeholder, columns=['Tc_factor', 'S_factor'])


#Plot the Samples
sns.kdeplot(df['Tc_factor'], label='Tc_factor', fill=True)
sns.kdeplot(df['S_factor'], label='S_factor', fill=True)
plt.xlabel('value')
plt.ylabel('Probability')
plt.title("Distribution")
plt.legend()
plt.show()
