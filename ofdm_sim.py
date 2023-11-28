import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import math
import csv


def theoreticalBER(EbNodB):
    return 0.5 * special.erfc(np.sqrt(pow(10, EbNodB / 10)))


def calculateBITno(theoreticalBER):
    return pow(10, -np.log10(theoreticalBER) + 2.2)


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
    mapping_table = {
        (0, 0): -1 - 1j,
        (0, 1): -1 + 1j,
        (1, 0): +1 - 1j,
        (1, 1): +1 + 1j,
    }
    symbols = np.array([mapping_table[tuple(pair)] for pair in bits_SP])
    # TODO: mindegyik x2 - 1 és akkor az amplitúdó
    return symbols


def IFFT(symbols_frequency):
    num_rows, num_cols = symbols_frequency.shape
    ifft_results = np.empty_like(symbols_frequency, dtype=complex)

    for i in range(num_rows):
        subarray = symbols_frequency[i]  # Access each subarray
        ifft_result = np.fft.ifft(subarray)  # Apply IFFT to the subarray
        ifft_results[i] = ifft_result
    signal_time = ifft_results
    return signal_time


def addCP(signal_time, CP=10):
    num_rows, num_cols = signal_time.shape
    signal_time_withCP = np.empty((num_rows, num_cols + CP), dtype=complex)

    for i in range(num_rows):
        prefix = np.array(signal_time[i, -CP:])
        signal_time_withCP[i] = np.concatenate((prefix, signal_time[i]))

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
    num_rows, num_cols = signal_time_withCP.shape
    signal_time = np.empty((num_rows, num_cols - CP), dtype=complex)

    for i in range(num_rows):
        signal_time[i] = np.array(signal_time_withCP[i, CP:])
    return signal_time


def FFT(signal_time):
    num_rows, num_cols = signal_time.shape
    fft_results = np.empty_like(signal_time, dtype=complex)

    for i in range(num_rows):
        subarray = signal_time[i]  # Access each subarray
        fft_result = np.fft.fft(subarray)  # Apply FFT to the subarray
        fft_results[i] = fft_result
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
    num_rows, num_cols = QAM.shape

    demapped_bits = np.empty((num_rows, num_cols * 2), dtype=int)
    for i in range(num_rows):
        returned = demap_one(QAM[i]).reshape(-1)
        demapped_bits[i] = returned
    return np.array(demapped_bits).reshape(-1, 2)

def demap_one(symbol):
    # print("QAM", symbol)
    # print("QAM type", type(symbol))

    demapping_table = {
        (-1 - 1j): (0, 0),
        (-1 + 1j): (0, 1),
        (+1 - 1j): (1, 0),
        (+1 + 1j): (1, 1),
    }
    # szimbólumok tömbje
    constellation = np.array([x for x in demapping_table.keys()])

    # a legközelebbi pontokhoz tartozó távolságok megkaphatók a komplex számok
    # különbségeként ami vektorokként szemléletes
    diff_vectors = symbol.reshape((-1, 1)) - constellation.reshape((1, -1))
    dists = abs(diff_vectors)

    # megkeressük a legkisebb távolságok indexeit
    const_index = dists.argmin(axis=1)

    # helyettesítjük a vett konstellációkat a "helyes" konstellációkkal
    decision = constellation[const_index]

    # a vett konstellációkat a bitpárokra cseréljük
    received = np.vstack([demapping_table[C] for C in decision])
    return received

def PS(bits_parallel):
    # bits = [b for a in bits_parallel for b in a]
    return bits_parallel.reshape((-1,))

def calculateBER(bits, bits_received):
    error = 0
    array_sent = np.array(bits)
    array_received = np.array(bits_received)
    for i in range(len(array_sent)):
        if array_sent[i] != array_received[i]:
            error += 1
    return error / len(bits)


carrierNo = 128
mu = 2
channelNoResponse = np.array([1])
channelResponse = np.array([math.sqrt(9), math.sqrt(1)])
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

    # print(data_entry)
    data_pairs.append(data_entry)
    print("SNRdB:", SNRdB, "Bit Error Rates:", BitErrorRate_nochannel, BitErrorRate_channel, BitErrorRate_equalized, BitErrorRate_estimated)

print(data_pairs)

# Specify the file name
file_name = 'output.csv'

# Writing the list to a CSV file
with open(file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_pairs)

print(f"The list has been written to '{file_name}' successfully.")



# TODO: IMSC pontokért vektorizálás