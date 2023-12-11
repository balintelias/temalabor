import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import math
import csv


def theoreticalBER(EbNodB):
    return 0.5 * special.erfc(np.sqrt(pow(10, EbNodB / 10)))


def calculateBITno(theoreticalBER):
    return pow(10, -np.log10(theoreticalBER) + 2)


def calculateSymbolNo(bitNo, mu):
    return math.ceil(bitNo / mu)


def calculateOFDMsymbolNo(symbolNo, carrierNo):
    return math.ceil(symbolNo / carrierNo)


def generateBITs(bitNo, carrierNo, mu):
    bitnumber = math.ceil(math.ceil(bitNo / mu) / carrierNo) * carrierNo * mu
    bits = np.random.binomial(n=1, p=0.5, size=bitnumber)
    return bits


def SP(bits):
    bits_SP = bits.reshape(-1, 2)
    return bits_SP


def mapping(bits_SP):
    symbols = bits_SP*2 - 1
    symbols = symbols.dot([1, 1j])
    return symbols


def IFFT(symbols_frequency):
    ifft_results = np.fft.ifft(symbols_frequency, axis=1)
    return ifft_results



def addCP(signal_time, CP=10):
    # Select the last 2 columns
    prefix = signal_time[:, -CP:]

    # Select all the original columns
    original = signal_time[:, :]

    # Concatenate the last 2 columns with the original columns
    signal_time_withCP = np.concatenate((prefix, original), axis=1)
    return signal_time_withCP


def channel(signal, SNRdB, channelResponse):
    # konvolúció
    output_signal = np.convolve(signal, channelResponse, mode="same")
    signal_power = np.mean(abs(output_signal**2))
    sigma2 = signal_power * 10 ** (-SNRdB / 10)  # zajteljesítmény

    # komplex zaj sigma2 teljesítménnyel, kétdimenziós normális eloszlás
    noise_real = np.sqrt(sigma2) / 2 * np.random.randn(*output_signal.shape)
    noise_imaginary = np.sqrt(sigma2) / 2 * 1j * np.random.randn(*output_signal.shape)
    noise = noise_real + noise_imaginary

    # return output_signal
    return output_signal + noise


def removeCP(signal_time_withCP, CP=10):
    signal_time = signal_time_withCP[:, CP:]
    return signal_time


def FFT(signal_time):
    fft_results = np.fft.fft(signal_time, axis=1)
    return fft_results

def equalize(signal, channelResponse):
    # ez így bénácska, de működik
    shape = signal.shape
    num_zeros = shape[1] - channelResponse.shape[0]
    if num_zeros > 0:
        channelResponse = np.pad(channelResponse, (0, num_zeros), 'constant')

    response_frequency = np.fft.fft(channelResponse)
    shape2 = response_frequency.shape
    equalized_signal = np.array([symbol / response_frequency for symbol in signal])
    return equalized_signal

def estimate(signal, pilot_symbol):
    # az átvitel a signal[0] és a pilot_symbol "hányadosa"
    # H = Y / U = signal[0] / pilot_symbol
    # U = Y / H
    transfer_function = signal[0] / pilot_symbol
    estimated_signal = np.array([symbol / transfer_function for symbol in signal])
    return estimated_signal


def demapping(QAM):
    demapping_table = {
        (-1 - 1j): (0, 0),
        (-1 + 1j): (0, 1),
        (+1 - 1j): (1, 0),
        (+1 + 1j): (1, 1),
    }
    constellation = np.array([x for x in demapping_table.keys()])

    num_rows, num_cols = QAM.shape
    # QAM = QAM.reshape((8, 128))

    # Call the modified demap_qpsk function
    demapped_bits = demap_qpsk(QAM, constellation, demapping_table)

    # Reshape the received bit pairs into the original shape of symbols
    demapped_bits = demapped_bits.reshape(-1, 2)
    return demapped_bits

def demap_qpsk(symbols, constellation, demapping_table):
    # Reshape symbols and constellation for broadcasting
    symbols = symbols.reshape(*symbols.shape, 1)
    constellation = constellation.reshape(1, 1, -1)

    # Calculate the distance from each received symbol to each constellation point
    dists = np.abs(symbols - constellation)

    # Find the index of the closest constellation point for each received symbol
    const_index = dists.argmin(axis=-1)

    # Replace the received symbols with the corresponding constellation points
    decision = constellation[0, 0, const_index]

    # Replace the constellation points with their corresponding bit pairs
    received = np.array([demapping_table[C] for C in decision.flatten()])#.reshape(*symbols.shape)

    return received


#TODO: 
def demap_one_v2(symbol):
    shape = symbol.shape
    real = symbol.real
    imaginary = symbol.imag
    # symbol = np.array([real, imaginary]).reshape(-1, 2) 
    # symbol = symbol.dot([1, 1/1j])
    real = np.sign(real)
    imaginary = np.sign(imaginary)
    symbol = np.array([real, imaginary]).reshape(-1, 2)
    # +1 / 2
    symbol = symbol + 1
    symbol = symbol / 2
    symbol = symbol - 0.5
    symbol = np.sign(symbol)
    return symbol

def PS(bits_parallel):
    # bits = [b for a in bits_parallel for b in a]
    return bits_parallel.reshape((-1,))

def calculateBER(bits, bits_received):
    error = np.sum(bits != bits_received)
    return error / len(bits)


carrierNo = 128
mu = 2
channelNoResponse = np.array([1])
channelResponse = np.array([math.sqrt(0.05), math.sqrt(0.95)])
data_pairs = []

for x in range(11):
    #TODO: mainnek ilyen függvény dictionaryvel, hogy egymás után lehessen futtatni
    SNRdB = x
    calculated_bitNo = calculateBITno(theoreticalBER(SNRdB))
    OFDMsymbolNo = calculateOFDMsymbolNo(
        calculateSymbolNo(calculated_bitNo, mu), carrierNo
    )
    bits = generateBITs(calculated_bitNo, carrierNo, mu)
    bits_SP = SP(bits)
    symbols = mapping(bits_SP)
    symbols_frequency = symbols.reshape(-1, carrierNo)
    signal_time = IFFT(symbols_frequency)
    signal_time_withCP = addCP(signal_time)
    OFDM_tx = signal_time_withCP.reshape(-1)
    data_entry = []
    
    # simulation without channel
    OFDM_rx = channel(OFDM_tx, SNRdB, channelNoResponse)
    OFDM_rx = OFDM_rx.reshape(-1, carrierNo + 10)
    OFDM_noCP = removeCP(OFDM_rx)
    OFDM_frequency = FFT(OFDM_noCP)
    OFDM_demapped = demapping(OFDM_frequency)
    bits_received = PS(OFDM_demapped)
    BitErrorRate_nochannel = calculateBER(bits, bits_received)
    data_entry.extend([SNRdB, BitErrorRate_nochannel])

    # simulation with channel
    OFDM_rx = channel(OFDM_tx, SNRdB, channelResponse)
    OFDM_rx = OFDM_rx.reshape(-1, carrierNo + 10)
    OFDM_noCP = removeCP(OFDM_rx)
    OFDM_frequency = FFT(OFDM_noCP)
    OFDM_demapped = demapping(OFDM_frequency)
    bits_received = PS(OFDM_demapped)
    BitErrorRate_channel = calculateBER(bits, bits_received)
    data_entry.extend([BitErrorRate_channel])

    #simulation with channel and equalization
    OFDM_rx = channel(OFDM_tx, SNRdB, channelResponse)
    OFDM_rx = OFDM_rx.reshape(-1, carrierNo + 10)
    OFDM_noCP = removeCP(OFDM_rx)
    OFDM_frequency = FFT(OFDM_noCP)
    OFDM_equalized = equalize(OFDM_frequency, channelResponse)
    OFDM_demapped = demapping(OFDM_equalized)
    bits_received = PS(OFDM_demapped)
    BitErrorRate_equalized = calculateBER(bits, bits_received)
    data_entry.extend([BitErrorRate_equalized])

    #simulation with channel and estimation
    OFDM_rx = channel(OFDM_tx, SNRdB, channelResponse)
    OFDM_rx = OFDM_rx.reshape(-1, carrierNo + 10)
    OFDM_noCP = removeCP(OFDM_rx)
    OFDM_frequency = FFT(OFDM_noCP)
    # egyszerűség kedvéért az első szimbólum ismert lesz a vevő számára,
    # egy preamble szimbólum hozáadásával lehetne ezt helyettesíteni
    pilot_symbol = symbols_frequency[0]
    OFDM_equalized = estimate(OFDM_frequency, pilot_symbol)
    OFDM_demapped = demapping(OFDM_equalized)
    bits_received = PS(OFDM_demapped)
    BitErrorRate_estimated = calculateBER(bits, bits_received)
    data_entry.extend([BitErrorRate_estimated])
    data_entry.extend([theoreticalBER(SNRdB)])

    data_pairs.append(data_entry)
    print("SNRdB:", SNRdB, "Bit Error Rates:", BitErrorRate_nochannel, BitErrorRate_channel, BitErrorRate_equalized, BitErrorRate_estimated, theoreticalBER(SNRdB))

print(data_pairs)

# Specify the file name
file_name = 'output.csv'

# Writing the list to a CSV file
with open(file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_pairs)

print(f"The list has been written to '{file_name}' successfully.")


