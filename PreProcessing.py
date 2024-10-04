from sktime.libs.vmdpy import VMD
from skimage.restoration import denoise_wavelet
import numpy as np
from scipy.signal import hilbert

def extract_vmd(Signal): # apply a wavelet to denoise a signal e then extract the third IMF of a VMD

    alpha = 1500       # moderate bandwidth constraint  
    tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
    K = 4            # 4 modes  
    DC = 0             # no DC part imposed  
    init = 1          # initialize omegas uniformly  
    tol = 1e-7

    f1 = denoise_wavelet(Signal, wavelet='haar', mode='soft', wavelet_levels=4, method='VisuShrink', rescale_sigma='True')
    u, u_hat, omega = VMD(f1, alpha, tau, K, DC, init, tol)
 
    return u[2]

def envelope(signal, rate = 25600): # apply an envelope to a signal and the extract the FFT from envelope
  amplitude_envelope = np.abs(hilbert(signal))
  fft_hilbert = np.fft.fftshift(2.0 * np.abs(np.fft.fft(amplitude_envelope) / np.size(np.fft.fft(amplitude_envelope))))
  freq_hilbert = np.fft.fftshift(np.fft.fftfreq(np.size(fft_hilbert), 1/rate))
  size =int(1 + len(fft_hilbert)/2)
  return freq_hilbert[size:] , fft_hilbert[size:]

def pre_processing(signal):
   aux1 = extract_vmd(signal)
   freq, env = envelope(aux1)
   return freq, env

