import numpy as np 
import matplotlib.pyplot as plt

SNRdB_range = SNRdB_range = np.arange(0, 11, 1)
# worst user SER
max_lld = np.array([1.81014706e-01, 1.42627451e-01, 1.03901961e-01, 7.13529412e-02,
    4.50392157e-02, 2.59117647e-02, 1.29460784e-02, 5.86764706e-03, 2.48529412e-03,
    7.54901961e-04, 2.59803922e-04])
det1_only = np.array([2.49034314e-01, 2.34901961e-01, 2.22014706e-01, 2.08284314e-01,
    1.93014706e-01, 1.76137255e-01, 1.58093137e-01, 1.42068627e-01, 1.23911765e-01,
    1.04004902e-01, 8.35147059e-02])
det1_det2 = np.array([1.99916667e-01, 1.61088235e-01, 1.24161765e-01, 8.81960784e-02,
    5.99362745e-02, 3.71715686e-02, 2.15735294e-02, 1.19558824e-02, 7.65686275e-03,
    1.09313725e-02, 2.12058824e-02])
det2_only = np.array([1.99710784e-01, 1.60617647e-01, 1.24627451e-01, 8.95343137e-02,
    5.95784314e-02, 3.64068627e-02, 2.07647059e-02, 1.04705882e-02, 4.57843137e-03,
    1.88725490e-03, 5.49019608e-04])

colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']

plt.plot(SNRdB_range, max_lld, color="tab:blue", marker="o", markersize=7, label='maximum likelihood')
plt.plot(SNRdB_range, det1_only, color="tab:green", marker="v", markersize=7, label='MMSE only')
plt.plot(SNRdB_range, det1_det2, color="tab:orange", marker="*", markersize=7, label='MMSE + NN')
plt.plot(SNRdB_range, det2_only, color="tab:red", marker="x", markersize=7, label='NN only')

plt.yscale('log')
plt.ylim(1e-4, 1)
plt.xlabel('SNR(dB)')
plt.ylabel('Symbol Error Rate')
plt.legend(loc='lower left', fontsize = 'x-small')
plt.grid(True)
plt.savefig('./MIMO_figures/MIMO_4to4_BPSK_SER_FixH_CH9_WorstUser.png')