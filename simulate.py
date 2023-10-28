import numpy as np
import matplotlib.pyplot as plt

# Az infokommunikáció tárgy tudásait felhasználva:
# Egyszerű OFDM rendszer szimulációja

"""
The following code is modified from this source:
https://dspillustrations.com/pages/posts/misc/python-ofdm-example.html
"""

"""
Specifikációk: QPSK moduláció 128 alvivővel
A QPSK moduláció megfeleltethető a 4QAM modulációnak, ami számomra szemléletesebb
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

def SP(bits):
    return bits.reshape((len(dataCarriers), mu))

def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])

def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = QAM_payload  # allocate the pilot subcarriers
    return symbol

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def channel(signal):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  # calculate noise power based on signal power and SNR
    
    # print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
    
    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)

def channelEstimate(OFDM_demod):
    pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
    Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values
    
    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase 
    # separately
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)
    
    plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
    plt.stem(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
    plt.plot(allCarriers, abs(Hest), label='Estimated channel via interpolation')
    plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
    plt.ylim(0,2)

    # plt.show()
    plt.savefig('estimatedChannel.png', bbox_inches='tight')
    plt.close()
    
    return Hest

def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized):
    return equalized[dataCarriers]

def Demapping(QAM):
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
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision

def PS(bits):
    return bits.reshape((-1,))

BER = []

for x in range(45):

    K = 128 # number of OFDM subcarriers


    P = 16 # number of pilot carriers per OFDM block
    pilotValue = 1+1j # The known value each pilot transmits

    allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

    pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.

    # For convenience of channel estimation, let's make the last carriers also be a pilot
    pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
    P = P+1

    # data carriers are all remaining carriers
    dataCarriers = np.delete(allCarriers, pilotCarriers)

    # print ("allCarriers:   %s" % allCarriers)
    # print ("pilotCarriers: %s" % pilotCarriers)
    # print ("dataCarriers:  %s" % dataCarriers)
    plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
    plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
    plt.legend()
    # plt.show()
    plt.savefig('carriers.png', bbox_inches='tight')
    plt.close()

    mu = 2 # bits per symbol (i.e. 4QAM / QPSK)
    payloadBits_per_OFDM = len(dataCarriers)*mu  # number of payload bits per OFDM symbol

    mapping_table = {
        (0,0) : -1-1j,
        (0,1) : -1+1j,
        (1,0) : +1-1j,
        (1,1) : +1+1j,
    }
    for b1 in [0, 1]:
        for b0 in [0, 1]:
            B = (b1, b0)
            Q = mapping_table[B]
            plt.plot(Q.real, Q.imag, 'bo')
            plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha='center')
            plt.xlabel('Imaginary Part (Q)')
            plt.ylabel('Real Part (I)')
            plt.title('QPSK or 4QAM Constellation')
    # plt.show()
    plt.savefig('constellation.png', bbox_inches='tight')
    plt.close()



    demapping_table = {v : k for k, v in mapping_table.items()}

    channelResponse = np.array([0.1])  # the impulse response of the wireless channel
    """
    Frekvenciafüggetlen átvitellel számolok
    """
    H_exact = np.fft.fft(channelResponse, K)
    plt.plot(allCarriers, abs(H_exact))
    # plt.show()
    plt.savefig('channelResponse.png', bbox_inches='tight')
    plt.close()

    SNRdb = x-20  # signal to noise-ratio in dB at the receiver 

    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
    # print ("Bits count: ", len(bits))
    # print ("First 20 bits: ", bits[:20])
    # print ("Mean of bits (should be around 0.5): ", np.mean(bits))


    bits_SP = SP(bits)
    # print ("First 5 bit groups")
    # print (bits_SP[:5,:])


    QAM = Mapping(bits_SP)
    # print ("First 5 QAM symbols and bits:")
    # print (bits_SP[:5,:])
    # print (QAM[:5])


    OFDM_data = OFDM_symbol(QAM)
    # print ("Number of OFDM carriers in frequency domain: ", len(OFDM_data))


    OFDM_time = IDFT(OFDM_data)
    # print ("Number of OFDM samples in time-domain before CP: ", len(OFDM_time))


    OFDM_TX = OFDM_time
    OFDM_RX = channel(OFDM_TX)
    plt.figure(figsize=(8,2))
    plt.plot(abs(OFDM_TX), label='TX signal')
    plt.plot(abs(OFDM_RX), label='RX signal')
    plt.legend(fontsize=10)
    plt.xlabel('Time'); plt.ylabel('$|x(t)|$');
    plt.grid(True);

# plt.show()
    plt.savefig('rxtx.png', bbox_inches='tight')
    plt.close()



    OFDM_demod = DFT(OFDM_RX)


    Hest = channelEstimate(OFDM_demod)


    equalized_Hest = equalize(OFDM_demod, Hest)


    QAM_est = get_payload(equalized_Hest)
    plt.plot(QAM_est.real, QAM_est.imag, 'bo');


    # plt.show()
    plt.savefig('receivedConstellation.png', bbox_inches='tight')
    plt.close()

    PS_est, hardDecision = Demapping(QAM_est)
    for qam, hard in zip(QAM_est, hardDecision):
        plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o');
        plt.plot(hardDecision.real, hardDecision.imag, 'ro')


    bits_est = PS(PS_est)

    print ("Obtained Bit error rate: ", np.sum(abs(bits-bits_est))/len(bits))

    # plt.show()
    plt.savefig('outputConstellation.png', bbox_inches='tight')
    plt.close()

    value = (SNRdb, np.sum(abs(bits-bits_est))/len(bits))
    BER.append(value)

print(BER)
plt.plot(*zip(*BER))
plt.show()

