import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('output.csv')

# Extract data pairs for plotting
SNR_ideal = list(zip(df.iloc[:, 0], df.iloc[:, 1]))
SNR_channel = list(zip(df.iloc[:, 0], df.iloc[:, 2]))
SNR_equalization = list(zip(df.iloc[:, 0], df.iloc[:, 3]))
SNR_estimation = list(zip(df.iloc[:, 0], df.iloc[:, 4]))
SNR_theorethical = list(zip(df.iloc[:, 0], df.iloc[:, 5]))

x1, y1 = zip(*SNR_ideal)
x2, y2 = zip(*SNR_channel)
x3, y3 = zip(*SNR_equalization)
x4, y4 = zip(*SNR_estimation)
x5, y5 = zip(*SNR_theorethical)

# ideal vs theoretical
plt.scatter(x1, y1)
plt.scatter(x5, y5)
plt.plot(x1, y1)
plt.plot(x5, y5)
plt.xlabel('SNR (in dB)')
plt.ylabel('Bit Error Rate')
plt.title('Bit Error Rate Simulation of an OFDM System with QPSK modulation')
plt.grid(True)
plt.legend(['Ideal Channel', 'Theoretical'])
plt.semilogy()
# plt.show()
plt.savefig('ideal_theoretical.png')
plt.close()

#ideal vs channel
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.plot(x1, y1)
plt.plot(x2, y2)
plt.xlabel('SNR (in dB)')
plt.ylabel('Bit Error Rate')
plt.title('Bit Error Rate Simulation of an OFDM System with QPSK modulation')
plt.grid(True)
plt.legend(['Ideal Channel', 'Channel'])
plt.semilogy()
# plt.show()
plt.savefig('ideal_channel.png')
plt.close()

#ideal vs equalization
plt.scatter(x1, y1)
plt.scatter(x3, y3)
plt.plot(x1, y1)
plt.plot(x3, y3)
plt.xlabel('SNR (in dB)')
plt.ylabel('Bit Error Rate')
plt.title('Bit Error Rate Simulation of an OFDM System with QPSK modulation')
plt.grid(True)
plt.legend(['Ideal Channel', 'Equalized Channel'])
plt.semilogy()
# plt.show()
plt.savefig('ideal_equalized.png')
plt.close()

#ideal vs estimation
plt.scatter(x1, y1)
plt.scatter(x4, y4)
plt.plot(x1, y1)
plt.plot(x4, y4)
plt.xlabel('SNR (in dB)')
plt.ylabel('Bit Error Rate')
plt.title('Bit Error Rate Simulation of an OFDM System with QPSK modulation')
plt.grid(True)
plt.legend(['Ideal Channel', 'Estimated Channel'])
plt.semilogy()
# plt.show()
plt.savefig('ideal_estimated.png')
plt.close()


# plt.scatter(x2, y2)
# plt.scatter(x3, y3)
# plt.scatter(x4, y4)
# plt.plot(x2, y2)
# plt.plot(x3, y3)
# plt.plot(x4, y4)
# plt.xlabel('SNR (in dB)')
# plt.ylabel('Bit Error Rate')
# plt.title('Bit Error Rate Simulation of an OFDM System with QPSK modulation')
# plt.grid(True)
# plt.legend(['Channel', 'Channel and equalized', 'Channel and estimated'])
# plt.semilogy()
# # plt.show()
# #plt.savefig()
# plt.close()
