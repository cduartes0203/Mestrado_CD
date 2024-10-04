import numpy as np
import math

def envelope_energy(signal):
    return sum(signal**2)

def envelope_band_energy(signal,freq, parts = 10):
    vec = []
    band = int(freq[-1]/parts)
    for i in range(parts):
        if i == 0:
            vec.append(envelope_energy(signal))
            vec.append(envelope_energy(signal[(freq>=i*band) & (freq<=(i+1)*band)]))
        else:
            vec.append(envelope_energy(signal[(freq>=i*band) & (freq<=(i+1)*band)]))
    return vec

def related_similarity(x, y):
    num, aux1, aux2 = 0, 0, 0
    for i in range(len(x)):
        num = num + (x[i] - np.mean(x)) * (y[i] - np.mean(y))
        aux1 = aux1 + (x[i] - np.mean(x)) **2 
        aux2 = aux2 + (y[i] - np.mean(y)) **2
    return abs(num)/math.sqrt(aux1*aux2)