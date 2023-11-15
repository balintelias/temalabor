import numpy as np
import matplotlib.pyplot as plt
from scipy import special

K = 128 #vivők száma
P = 1 # 1 pilot vivő
mu = 2 # 2 bit/szimbólum
carrierNo = K - P

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
        (0,0) : -1-1j,
        (0,1) : -1+1j,
        (1,0) : +1-1j,
        (1,1) : +1+1j,
    }
    symbols = [mapping_table[pair] for pair in bits_SP]
    return symbols

# Inverz FFT: 
# A frekvenciatartományban lévő szimbólumokból időtartománybeli jelet képez
def IFFT(symbols):
    return np.fft.ifft(symbols)

# Átviteli csatorna
# megadott teljesítményű addítív fehér Gauss-zajjal
def channel(signal, SNRdb, channelResponse):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10) 

    # Generate complex noise with given variance
    noise = np.sqrt(sigma2)/2 * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    # Itt kellett kivenni a /2-t a gyökből
    # TODO: ellenőrizni a jel és a zaj teljesítményét
    return convolved + noise


def FFT(ofdm_parallel):
    return np.fft.fft(ofdm_parallel)


def demapping(QAM):
    demapping_table = {
         (-1-1j) : (0,0),
         (-1+1j) : (0,1),
         (+1-1j) : (1,0),
         (+1+1j) : (1,1),
    }
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])
    
    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
    
    # for each element in QAM, choose the index in constellation 
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    # return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision
    received = np.vstack([demapping_table[C] for C in hardDecision]) 
    return received

# bit
def PS(bits_parallel):
    # bits = [b for a in bits_parallel for b in a]
    return bits_parallel.reshape((-1,))

# ber
def calculate_BER(OFDM_demapped, bits):
    error = 0
    for i in range(len(bits)):
        if bits[i] != OFDM_demapped[i]:
            error += 1
    return error/len(bits)


simulated_BER = []

for x in range(11):
    SNRdB = x
    payloadBits_per_OFDM = mu * (K - P)
    bitsToSimulate = calculateBITno(theoreticalBER(SNRdB))
    numberOfSymbols = bitsToSimulate / payloadBits_per_OFDM + 1
    pairs = []
    channelResponse = np.array([1])
    for symbol in range(int(numberOfSymbols)):
        # breakpoint()
        bits = generateBITs(payloadBits_per_OFDM)
        bits_SP = SP(bits)
        symbols = mapping(bits_SP)
        ofdm_time = IFFT(symbols)
        ofdm_received = channel(ofdm_time, SNRdB, [1])
        ofdm_demod = FFT(ofdm_received)
        demapped = demapping(ofdm_demod)
        bits_received = PS(demapped)
        calculated_BER = calculate_BER(bits_received, bits)
        pairs.append((SNRdB, calculated_BER))
        print ("SNR: ", x, "Symbols:", symbol, "/", numberOfSymbols,  " Obtained Bit error rate: ", calculated_BER)

    BERs = []
    for item in pairs:
        BERs.append(item[1])
    # print(BERs)
    meanBER = np.mean(BERs)
    simulated_BER.append((x, meanBER))
    print("SNR: ", x,  " Mean BER: ", meanBER)

print(simulated_BER)
plt.plot(*zip(*simulated_BER))
plt.semilogy()
#plt.show()
plt.savefig('simulatedBER2.png', bbox_inches='tight')
plt.close()