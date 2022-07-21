import numpy as np 
import matplotlib.pyplot as plt

# SER:  limegreen, lightgreen, green
# BER:  deepskyblue, skyblue, steelblue
# CRC: tab:blue, tab:green, tab:orange, tab:red

SNRdB_range = SNRdB_range = np.arange(0, 11, 1)
CRC_optimal = np.array([0. , 0.007, 0.114, 0.419, 0.78, 0.965, 0.991, 1. , 1. , 1. , 1.])
CRC_det1 = np.array([0. ,  0.001, 0. , 0.004, 0.004, 0.004, 0.004, 0.016, 0.033, 0.115, 0.3 ])
CRC_det1_det2 = np.array([0. , 0.005, 0.027, 0.223, 0.584, 0.881, 0.976, 0.996, 1., 1., 1.])
CRC_det2 = np.array([0.002, 0.002, 0.025, 0.23, 0.613, 0.879, 0.978, 0.997, 1., 0.999, 1.])

plt.plot(SNRdB_range, CRC_optimal, color="tab:blue", marker="o", markersize=7, label='maximum likelihood')
plt.plot(SNRdB_range, CRC_det1, color="tab:green", marker="v", markersize=7, label='MMSE only')
plt.plot(SNRdB_range, CRC_det1_det2, color="tab:orange", marker="*", markersize=7, label='MMSE + NN')
plt.plot(SNRdB_range, CRC_det2, color="tab:red", marker="x", markersize=7, label='NN only')
#plt.yscale('log')
plt.ylim(0, 1.1)

plt.xlabel('SNR(dB)')
plt.ylabel('CRC pass rate')
plt.legend(loc='upper left', fontsize = 'x-small')
plt.grid(True)
plt.savefig('./MIMO_figures/MIMO_4to4_BPSK_CRC_FixH_MMSE_CH9.png')