# BER = 1/2 (erfc(sqrt(Eb/No)))
# erfc (x) = 2/sqrt(pi) * integral from x to infinity of e^(-t^2) dt

import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def BER(EbNo):
    return 0.5 * special.erfc(np.sqrt(EbNo))

def bitNO(BER):
    return pow(10, -np.log10(BER)+2)

array = []
for x in range(10):
    array.append((x, BER(x), bitNO(BER(x))))

print(array)
x = np.linspace(0, 10, 100)
plt.plot(x, BER(x))
plt.xlabel('$SNR$')
plt.ylabel('$BER(SNR)$')
plt.yscale('log')
plt.show()