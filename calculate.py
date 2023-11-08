# BER = 1/2 (erfc(sqrt(Eb/No)))
# erfc (x) = 2/sqrt(pi) * integral from x to infinity of e^(-t^2) dt

import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import math

def BER(EbNo):
    return 0.5 * math.erfc(math.sqrt(pow(10, EbNo/10)))

def bitNO(BER):
    return pow(10, -np.log10(BER)+2)

array = []
for x in range(10):
    array.append((x, BER(x), bitNO(BER(x))))

print(array)
x = np.array([0,1,2,3,4,5,6,7,8,9])
BERs = []
for number in x:
    BERs.append(BER(number))
# plt.plot(x, BERs)
plt.xlabel('$SNR$')
plt.ylabel('$BER(SNR)$')
plt.semilogy(x, BERs)
plt.show()