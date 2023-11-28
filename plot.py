import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('output.csv')

# Extract data pairs for plotting
first_second_pairs = list(zip(df.iloc[:, 0], df.iloc[:, 1]))
first_third_pairs = list(zip(df.iloc[:, 0], df.iloc[:, 2]))
first_fourth_pairs = list(zip(df.iloc[:, 0], df.iloc[:, 3]))
first_fifth_pairs = list(zip(df.iloc[:, 0], df.iloc[:, 4]))

x1, y1 = zip(*first_second_pairs)
x2, y2 = zip(*first_third_pairs)
x3, y3 = zip(*first_fourth_pairs)
x4, y4 = zip(*first_fifth_pairs)
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.scatter(x3, y3)
plt.scatter(x4, y4)
plt.plot(x1, y1)
plt.plot(x2, y2)
plt.plot(x3, y3)
plt.plot(x4, y4)
plt.xlabel('SNR (in dB)')
plt.ylabel('Bit Error Rate')
plt.title('Bit Error Rate Simulation of an OFDM System with QPSK modulation')
plt.grid(True)
plt.legend(['No channel', 'Channel', 'Channel and equalized', 'Channel and estimated'])
plt.semilogy()
plt.show()
