import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import math


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
    mapping_table = {
        (0, 0): -1 - 1j,
        (0, 1): -1 + 1j,
        (1, 0): +1 - 1j,
        (1, 1): +1 + 1j,
    }
    symbols = np.array([mapping_table[tuple(pair)] for pair in bits_SP])
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
        type1 = type(prefix)
        type2 = type(signal_time[i])
        type3 = type(signal_time_withCP)
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

def equalize(signal, channelResponse):
    response_frequency = np.fft.fft(channelResponse)
    inverse_response = np.fft.ifft(1 / response_frequency)
    output_signal = np.convolve(signal, inverse_response, mode="same")
    return output_signal


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
    # signal_time = fft_results
    return fft_results


def demapping(QAM):
    num_rows, num_cols = QAM.shape

    demapped_bits = np.empty((num_rows, num_cols * 2), dtype=int)
    for i in range(num_rows):
        returned = demap_one(QAM[i]).reshape(-1)
        shape1 = returned.shape
        shape2 = demapped_bits[i].shape
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
# channelResponse = np.array([1])
channelResponse = np.array([math.sqrt(9), math.sqrt(1)])
data_pairs = []

for x in range(11):
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
    OFDM_rx = channel(OFDM_tx, SNRdB, channelResponse)
    OFDM_equalized = equalize(OFDM_rx, channelResponse)
    # OFDM_equalized = OFDM_rx
    OFDM_equalized = OFDM_equalized.reshape(-1, carrierNo + 10)
    OFDM_noCP = removeCP(OFDM_equalized)
    OFDM_frequency = FFT(OFDM_noCP)
    OFDM_demapped = demapping(OFDM_frequency)
    bits_received = PS(OFDM_demapped)
    BitErrorRate = calculateBER(bits, bits_received)
    data_pairs.append((SNRdB, BitErrorRate))
    print("SNRdB: ", SNRdB, "Bit Error Rate: ", BitErrorRate)

print(data_pairs)
plt.plot(*zip(*data_pairs))
plt.semilogy()
plt.xlabel("SNR (in dB)")
plt.ylabel("Bit Error Rate")
plt.title("Bit Error Rate Simulation of an OFDM System with QPSK modulation")
plt.suptitle("Frequency Dependent Channel, Equalized")
plt.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
plt.savefig("simulatedBER_equalized.png", bbox_inches="tight")
plt.show()
plt.close()