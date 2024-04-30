import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(r"\DATA\Event_rainfall.csv")

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set the index of the DataFrame to the 'Date' column
df.set_index('Date', inplace=True)

# Create a bar plot
ax = df['Rainfall'].plot(kind='bar', figsize=(16,9),color='#48737A')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=0)

# Set the x-axis labels
x_labels = [time[:-3] for time in df.index.strftime('%H:%M')]
plt.yticks(fontsize=16)
plt.xticks(range(len(df)), x_labels, fontsize=16)

# Add a title and axis labels
plt.title('Rainfall Data',fontsize=25)
plt.xlabel('Time (Hourly)',fontsize=20)
plt.ylabel('Rainfall (mm)',fontsize=20)

# Display the plot
plt.tight_layout()
plt.show()


#Streamflow:

df1= pd.read_csv(r"\DATA\Event_streamflow_node1.csv")

# Convert 'Date' column to datetime
df1['Date'] = pd.to_datetime(df1['Date'])
df1.set_index('Date', inplace=True)
# Plotting
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(df1['Daily_Streamflow_Node1'],  label='Node 1',  marker='o', linewidth=3)
ax.plot(df1['Daily_Streamflow_Node28'], label='Node 28', marker='o',linewidth=3)
ax.plot(df1['Daily_Streamflow_Node54'], label='Node 54', marker='o', linewidth=3)
ax.plot(df1['Daily_Streamflow_Node59'], label='Node 59', marker='o', linewidth=3)

ax.set_xlabel('Date', fontsize=20)
ax.set_ylabel('Daily Streamflow', fontsize=20)
ax.set_title('Daily Streamflow for Different Nodes', fontsize=25)
# plt.legend(fontsize=16)
# plt.grid(True)
# plt.yticks(fontsize=16)

# plt.xticks(rotation=45, ha='right', fontsize=16)

# plt.tight_layout()
# plt.show()
# Rotate and adjust x-axis labels
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # Format x-axis labels to show month-day
plt.gcf().autofmt_xdate()  # Automatically rotate and align the date labels
plt.xticks(rotation=0,fontsize=16,ha='right')  # Rotate x-axis labels by 45 degrees and align horizontally

# Add grid and adjust tick font sizes
ax.legend(fontsize=18)
ax.grid(True)
ax.tick_params(axis='y', labelsize=16)
plt.tight_layout()
plt.show()
