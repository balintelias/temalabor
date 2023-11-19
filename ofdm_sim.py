import numpy as np
import matplotlib.pyplot as plt
from scipy import special


def theoreticalBER(EbNo):
    return 0.5 * special.erfc(np.sqrt(EbNo))


def calculateBITno(theoreticalBER):
    return pow(10, -np.log10(theoreticalBER) + 2)


def generateBITs(bitNo):
    bitnumber = bitNo
    bits = np.random.binomial(n=1, p=0.5, size=bitnumber)
    return bits


# S/P átalakító, bitcsoportokat csinál a bitsorozatból
def SP(bits):
    bits_SP = [(bits[i], bits[i + 1]) for i in range(0, len(bits), 2)]
    return bits_SP


# QPSK szimbólumokat képez a bitpárokból
def mapping(bits_SP):
    mapping_table = {
        (0, 0): -1 - 1j,
        (0, 1): -1 + 1j,
        (1, 0): +1 - 1j,
        (1, 1): +1 + 1j,
    }
    symbols = [mapping_table[pair] for pair in bits_SP]
    return symbols


# Inverz FFT:
# A frekvenciatartományban lévő szimbólumokból időtartománybeli jelet képez
def IFFT(symbols):
    return np.fft.ifft(symbols)


# Átviteli csatorna
# megadott teljesítményű addítív fehér Gauss-zajjal
def channel(signal, SNRdB, channelResponse):
    # konvolúció
    output_signal = np.convolve(signal, channelResponse, mode="same")
    signal_power = np.mean(abs(output_signal**2))
    sigma2 = signal_power * 10 ** (-SNRdB / 10)  # zajteljesítmény

    # komplex zaj sigma2 teljesítménnyel, kétdimenziós normális eloszlás
    noise_real = np.sqrt(sigma2) / 2 * np.random.randn(*output_signal.shape)
    noise_imaginary = np.sqrt(sigma2) / 2 * 1j * np.random.randn(*output_signal.shape)
    noise = noise_real + noise_imaginary
    return output_signal + noise


def FFT(ofdm_parallel):
    return np.fft.fft(ofdm_parallel)


def demapping(QAM):
    demapping_table = {
        (-1 - 1j): (0, 0),
        (-1 + 1j): (0, 1),
        (+1 - 1j): (1, 0),
        (+1 + 1j): (1, 1),
    }
    # szimbólumok tömbje
    constellation = np.array([x for x in demapping_table.keys()])

    # calculate distance of each RX point to each possible point
    # a legközelebbi pontokhoz tartozó távolságok megkaphatók a komplex számok
    # különbségeként ami vektorokként szemléletes
    diff_vectors = QAM.reshape((-1, 1)) - constellation.reshape((1, -1))
    dists = abs(diff_vectors)

    # megkeressük a legkisebb távolságok indexeit
    const_index = dists.argmin(axis=1)

    # helyettesítjük a vett konstellációkat a "helyes" konstellációkkal
    decision = constellation[const_index]

    # a vett konstellációkat a bitpárokra cseréljük
    received = np.vstack([demapping_table[C] for C in decision])
    return received


# bit
def PS(bits_parallel):
    # bits = [b for a in bits_parallel for b in a]
    return bits_parallel.reshape((-1,))


# ber
def calculate_BER(OFDM_demapped, bits):
    error = 0
    OFDM_1D = np.concatenate(OFDM_demapped)
    len1 = len(OFDM_1D)
    len2 = len(bits)
    array_sent = np.array(bits)
    array_received = np.array(OFDM_1D)
    for i in range(len(array_sent)):
        if array_sent[i] != array_received[i]:
            error += 1
    return error / len(bits)


K = 128  # vivők száma
P = 0  # 1 pilot vivő
mu = 2  # 2 bit/szimbólum
carrierNo = K - P

simulated_BER = []

for x in range(11):  # for simulation
    # for x in range(6): # for testing
    SNRdB = x
    payloadBits_per_OFDM = mu * (K - P)
    bitsToSimulate = calculateBITno(theoreticalBER(SNRdB))
    numberOfSymbols = int(bitsToSimulate / payloadBits_per_OFDM + 10)

    # channelResponse = np.array([np.sqrt(0.6), np.sqrt(0.3), np.sqrt(0.1)]) # négyzetösszeg legyen
    channelResponse = np.array([1])  # for testing

    pairs = []
    signal_time = []

    # a generálást egészen addig amíg időtartományba nem transzformálunk, továbbra is for ciklussal
    ofdm_time = []
    bits_sum = []
    for symbol in range(numberOfSymbols):
        bits = generateBITs(payloadBits_per_OFDM)
        bits_SP = SP(bits)
        symbols = mapping(bits_SP)
        symbol_time = IFFT(symbols)
        ofdm_time.extend(symbol_time)
        bits_sum.extend(bits)
    # ezzel megvan az összes szimbólum egyesített időfüggvénye
    len_time = len(ofdm_time)
    ofdm_received = channel(
        ofdm_time, SNRdB, channelResponse
    )  # ehhez nem kell nyúlni sztem
    len_received = len(ofdm_received)
    # ezután a vett jelet kell "szétvágni" szimbólumokra
    received_symbols = ofdm_received.reshape((-1, K))  # vajon jó?
    len_received_symbols = len(received_symbols)
    # majd a szimbólumokat vissza kell transzformálni frekvenciatartományba
    received_symbols_freq = [FFT(symbol) for symbol in received_symbols]
    len_received_symbols_freq = len(received_symbols_freq)
    # végül a szimbólumokat demodulálni kell
    demapped = [demapping(symbol_freq) for symbol_freq in received_symbols_freq]
    len_demapped = len(demapped)
    # és a bitpárokat vissza kell alakítani bitekké
    bits_received = [PS(demapped_symbol) for demapped_symbol in demapped]
    calculated_BER = calculate_BER(bits_received, bits_sum)
    print("SNR: ", x, " Obtained Bit error rate: ", calculated_BER)
    simulated_BER.append((x, calculated_BER))

print(simulated_BER)
plt.plot(*zip(*simulated_BER))
plt.semilogy()
# plt.show()
plt.xlabel("SNR (in dB)")
plt.ylabel("Bit Error Rate")
plt.title("Bit Error Rate Simulation of an OFDM System with QPSK modulation")
plt.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
plt.savefig("simulatedBER_sajat.png", bbox_inches="tight")

plt.close()

# TODO:
# FIXME:

# IMSC pontokért vektorizált mátrixos _izé_

# TODO:
# CP megírása:
# CP hossza > h hossza (csatorna impulzusválasza)

# TODO:
# H ismeretében H inverzével csatornakompenzálás
