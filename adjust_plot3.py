import numpy as np 
import matplotlib.pyplot as plt

SNRdB_range = SNRdB_range = np.arange(0, 11, 1)
# best user BER
max_lld = np.array([3.7900e-03, 7.5000e-04, 1.6000e-04, 2.0000e-05, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan, np.nan])
det1_only = np.array([4.7600e-03, 1.0600e-03, 5.0000e-05, 2.0000e-05, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan, np.nan])
det1_det2 = np.array([1.3260e-02, 4.3700e-03, 1.0800e-03, 1.8000e-04, 4.0000e-05, np.nan,
    np.nan, np.nan, np.nan, np.nan, np.nan])
det2_only = np.array([1.2240e-02, 4.6000e-03, 1.2800e-03, 2.5000e-04, 1.0000e-05, 2.0000e-05,
    np.nan, np.nan, np.nan, np.nan, np.nan])

colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']

plt.plot(SNRdB_range, max_lld, color="tab:blue", marker="o", markersize=7, label='maximum likelihood')
plt.plot(SNRdB_range, det1_only, color="tab:green", marker="v", markersize=7, label='MMSE only')
plt.plot(SNRdB_range, det1_det2, color="tab:orange", marker="*", markersize=7, label='MMSE + NN')
plt.plot(SNRdB_range, det2_only, color="tab:red", marker="x", markersize=7, label='NN only')

plt.yscale('log')
plt.xlim(-0.5, 11)
plt.ylim(1e-6, 1e-1)
plt.xlabel('SNR(dB)')
plt.ylabel('Bit Error Rate')
plt.legend(loc='lower right', fontsize = 'x-small')
plt.grid(True)
plt.savefig('./MIMO_figures/MIMO_4to4_BPSK_BER_FixH_CH9_BestUser.png')