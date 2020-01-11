import os
import numpy as np
import random

s_freq = 1000 #Sampling freq
beta = 1. #Fréquence des saccades
orr_mean_energy = 1000000 #Mean energy of the full signal
trend_mean_energy = 700000 #Mean energy of the trend
detrend_mean_energy = 200000 #Mean evergy of the detrended signal
sigma_noise = 3 #Level of noise
dt_sigm = 0.1

def jerk(t, curv=0):
    if t < 0.9:
        return curv*(t/0.9)**2 + (1-curv)*(t/0.9)-0.5
    else : 
        return 1-(t-0.9)/(1-0.9)-0.5

def gen_signal(sig_size=5000, nyst_type="pendular",s_freq=1000,  nyst_freq=4, trend_mean_energy=700000, orr_mean_energy=1000000, detrend_mean_energy=200000, dt_sigm=0.1, curv=0, sigma_noise=3):
    test = np.random.randn(sig_size) + np.random.randn(sig_size) * 1j
    test = test * (np.arange(sig_size)<2)
    signal = np.real(np.fft.ifft(test))
    signal = np.sqrt(trend_mean_energy) * signal /np.sqrt((np.sum(signal**2)/signal.shape[0]))
    last_sacc = 0
    saccs = []
    while last_sacc < sig_size:
        i = np.random.exponential(2000)
        start = last_sacc + i 
        sign = 2 * np.random.binomial(1,0.5)-1
        end = start+100
        amp = 1500 + 500 * np.random.randn()
        last_sacc = end
        saccs.append((start, end, sign, amp))
    saccsignal = np.zeros(sig_size)
    for i in range(sig_size):
        for start, end, sign, amp in saccs:
            if start < i :
                saccsignal[i]+= sign * amp * (1/(1+np.exp(-1.2*dt_sigm*(i-start-50))))
    signal = signal + saccsignal
    signal = np.sqrt(trend_mean_energy) * signal /np.sqrt((np.sum(signal**2)/signal.shape[0]))
    signal += sigma_noise * np.random.randn(sig_size)
    if nyst_type=="pendular":
        nyst = np.cos(2* np.pi * np.arange(sig_size)*nyst_freq/s_freq)
        nyst = np.sqrt(detrend_mean_energy) * nyst /np.sqrt((np.sum(nyst**2)/signal.shape[0]))/5
    elif nyst_type=="jerk_sf_up":
        nyst = np.array([jerk((k*nyst_freq%s_freq)/s_freq,curv) for k in range(sig_size)])
        nyst = np.sqrt(detrend_mean_energy) * nyst /np.sqrt((np.sum(nyst**2)/signal.shape[0]))/5
    elif nyst_type=="jerk_sf_down":
        nyst = np.array([jerk((k*nyst_freq%s_freq)/s_freq,curv) for k in range(sig_size)])
        nyst = -np.sqrt(detrend_mean_energy) * nyst /np.sqrt((np.sum(nyst**2)/signal.shape[0]))/5
    elif nyst_type=="jerk_fs_up":
        nyst = np.array([jerk((k*nyst_freq%s_freq)/s_freq,curv) for k in range(sig_size)])[::-1]
        nyst = np.sqrt(detrend_mean_energy) * nyst /np.sqrt((np.sum(nyst**2)/signal.shape[0]))/5
    elif nyst_type=="jerk_fs_down":
        nyst = np.array([jerk((k*nyst_freq%s_freq)/s_freq,curv) for k in range(sig_size)])[::-1]
        nyst = -np.sqrt(detrend_mean_energy) * nyst /np.sqrt((np.sum(nyst**2)/signal.shape[0]))/5
    return signal, nyst

SIZE=5000
NUM_SAMPLES=500
DATASET_TREND=np.zeros((NUM_SAMPLES, SIZE))
DATASET_NYST=np.zeros((NUM_SAMPLES, SIZE))
for i in range(NUM_SAMPLES):
    nyst_type = random.choice(["pendular", "pendular", "pendular", "pendular","jerk_sf_up", "jerk_sf_down", "jerk_fs_up", "jerk_fs_down"])
    DATASET_TREND[i], DATASET_NYST[i] = gen_signal(sig_size=SIZE, 
                                                    nyst_type=nyst_type, 
                                                    s_freq=1000,  
                                                    nyst_freq=max(1,min(4+2*np.random.randn(), 6)), 
                                                    trend_mean_energy=700000 + 10000*np.random.randn(), 
                                                    orr_mean_energy=5000000+500000*np.random.randn(), 
                                                    detrend_mean_energy=1000000+100000*np.random.randn(), 
                                                    dt_sigm=0.1, 
                                                    curv=max(0,min(0.5+0.25*np.random.randn(), 1)), 
                                                    sigma_noise=3)
np.save("trends", DATASET_TREND)
np.save("nysts", DATASET_NYST)