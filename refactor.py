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
    response_zeros = np.zeros_like(signal)
    for i in range(len(channelResponse)):
        response_zeros[i] = channelResponse[i]
    response_frequency = np.fft.fft(response_zeros)
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
    # print("bits:", bits)
    # print("bits type:", type(bits))
    # print("bits size:", bits.size)
    bits_SP = SP(bits)
    # print("bits:", bits_SP)
    # print("bits type:", type(bits_SP))
    # print("bits size:", bits_SP.size)
    symbols = mapping(bits_SP)
    # print("symbols:", symbols)
    # print("symbols type:", type(symbols))
    # print("symbols size:", symbols.size)OFDM_rx
    symbols_frequency = symbols.reshape(-1, carrierNo)
    # print("symbols_frequency:", symbols_frequency)
    # print("symbols_frequency type:", type(symbols_frequency))
    # print("symbols_frequency size:", symbols_frequency.size)
    signal_time = IFFT(symbols_frequency)
    # print("signal_time:", signal_time)
    # print("signal_time type:", type(signal_time))
    # print("signal_time size:", signal_time.size)
    # print("signal_time shape:", signal_time.shape)
    signal_time_withCP = addCP(signal_time)
    # print("signal_time:", signal_time_withCP)
    # print("signal_time type:", type(signal_time_withCP))
    # print("signal_time size:", signal_time_withCP.size)
    # print("signal_time shape:", signal_time_withCP.shape)
    OFDM_tx = signal_time_withCP.reshape(-1)
    # print("OFDM_tx:", OFDM_tx)
    # print("OFDM_tx type:", type(OFDM_tx))
    # print("OFDM_tx size:", OFDM_tx.size)
    # print("OFDM_tx shape:", OFDM_tx.shape)
    OFDM_rx = channel(OFDM_tx, SNRdB, channelResponse)
    # print("OFDM_rx:", OFDM_rx)
    # print("OFDM_rx type:", type(OFDM_rx))
    # print("OFDM_rx size:", OFDM_rx.size)
    # print("OFDM_rx shape:", OFDM_rx.shape)
    OFDM_equalized = equalize(OFDM_rx, channelResponse)
    OFDM_equalized = OFDM_equalized.reshape(-1, carrierNo + 10)
    # print("OFDM_rx:", OFDM_rx)
    # print("OFDM_rx type:", type(OFDM_rx))
    # print("OFDM_rx size:", OFDM_rx.size)
    # print("OFDM_rx shape:", OFDM_rx.shape)
    OFDM_noCP = removeCP(OFDM_equalized)
    # print("OFDM_noCP:", OFDM_noCP)
    # print("OFDM_noCP type:", type(OFDM_noCP))
    # print("OFDM_noCP size:", OFDM_noCP.size)
    # print("OFDM_noCP shape:", OFDM_noCP.shape)
    OFDM_frequency = FFT(OFDM_noCP)
    # print("OFDM_frequency:", OFDM_frequency)
    # print("OFDM_frequency type:", type(OFDM_frequency))
    # print("OFDM_frequency size:", OFDM_frequency.size)
    # print("OFDM_frequency shape:", OFDM_frequency.shape)
    OFDM_demapped = demapping(OFDM_frequency)
    # print("OFDM_demapped:", OFDM_demapped)
    # print("OFDM_demapped type:", type(OFDM_demapped))
    # print("OFDM_demapped size:", OFDM_demapped.size)
    # print("OFDM_demapped shape:", OFDM_demapped.shape)
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
# plt.text(s='Bit Error Rate Simulation of an OFDM System with QPSK modulation', x=0.5, y=1.00, fontsize=18, ha='center', va='center')
# plt.text(s='Frequency Independent Channel', x=0.5, y=0.95, fontsize=12, ha='center', va='center')
plt.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
plt.savefig("simulatedBER_frekifüggő.png", bbox_inches="tight")
# plt.show()
plt.close()