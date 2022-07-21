import numpy as np 
import torch
from torch import nn
import time
from CommPy.utils import *
from CommPy.modulation import QAMModem, PSKModem
import CommPy.channelcoding.convcode as cc
from CommPy.CRC import CRC_encode, CRC_decode
from CommPy.equalizer import pseudo_inverse
from initialization import BPSK_optimal
from networks.FCNet import BinaryClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

""" Tx Rx settings """
m = 100       # num of Rx antennas, or the num of symbols within one frame at the Rx
n = 100       # num of Tx antennas, or the num of symbols within one frame at the Tx
block_size = 10
num_blocks = 100
""" Modulation setting """
mod = 2     # BPSK
bit_width = int(np.log2(mod))

""" Create Modem """
Tx = PSKModem(mod)
constellation = Tx.constellation
# all symbols uniform distribution at first
lld_k_Initial = np.log(1/mod * np.ones(mod)) 

""" learning module setting """
model = BinaryClassifier()
ckpt_load_path = './checkpoints/BPSK_Tx100_Rx100_FlatFading_bound5e-1/epoch_1.pt'
# do not consider cuda currently
model.load_state_dict(torch.load(ckpt_load_path))
model.eval()

""" Channel Coding setting """
# convolutional codes
generator_matrix = np.array([[5, 7]])  # generator branches
trellis = cc.Trellis(np.array([mod]), generator_matrix) # Trellis structure
rate = 1/2    # code rate
constraint_len = 7        # constraint length
num_delay = np.array([constraint_len - 1])  # number of delay elements
# Viterbi decoder parameters
tb_depth = 5*(num_delay.sum() + 1)  # traceback depth

""" Channel Setting """
H_mean = 0  # flat fading
H_var = 1
SNRdB_range = np.arange(0, 18, 2)
# calculate H.shape
tx_sample = np.random.randint(0, 2, (bit_width*n, ))
tx_coded = cc.conv_encode(tx_sample, trellis)
H_width = int(tx_coded.shape[0] / bit_width)
rx_sample = np.random.randint(0, 2, (bit_width*m, ))
rx_coded = cc.conv_encode(rx_sample, trellis)
H_height = int(rx_coded.shape[0] / bit_width)

""" CRC setting """
CRC_key = '100000111' # CRC-8 polynomial

signal_power = 1
BER_decoded_main = np.zeros(SNRdB_range.shape) # calculate after decoding
BER_uncoded_main = np.zeros(SNRdB_range.shape) # calculate after demodulation
pass_rate = np.zeros(SNRdB_range.shape)        # calculate after CRC

""" MAIN LOOPS """
for dd in range(0, SNRdB_range.shape[0]):
    start_time = time.time()

    # in each SNRdB
    var_noise = signal_power * H_var * np.power(10, -0.1*SNRdB_range[dd])
    BER_decoded_block = np.zeros(num_blocks)
    BER_uncoded_block = np.zeros(num_blocks)
    fail_num = 0

    for bb in tqdm(range(0, num_blocks)):
        # in each block, H is the same
        H_real = H_mean + np.sqrt(0.5 * H_var)*np.random.randn(H_height, H_width)
        #H_real = np.zeros((H_height, H_width)) + 1
        H_imag = H_mean + np.sqrt(0.5 * H_var)*np.random.randn(H_height, H_width)
        #H_imag = np.zeros((H_height, H_width))
        H = H_real + 1j * H_imag
        bit_dataset = np.random.randint(0, 2, (block_size, bit_width*n))

        BER_decoded_vector = np.zeros(block_size)
        BER_uncoded_vector = np.zeros(block_size)

        for jj in range(0, block_size):
            # for each message
            message_bits = bit_dataset[jj]
            remainder, codeword = CRC_encode(message_bits, CRC_key)
            coded_bits = cc.conv_encode(message_bits, trellis)
            x_indices, x_symbols = Tx.modulate(coded_bits)
            noise_real = 0. + np.sqrt(0.5 * var_noise)*np.random.randn(H_height)
            noise_imag = 0. + np.sqrt(0.5 * var_noise)*np.random.randn(H_height)
            y_symbols = np.matmul(H, x_symbols) + (noise_real + 1j * noise_imag)
            #y_symbols = x_symbols + (noise_real + 1j * noise_imag)

            # estimation by zero-forcing then demodulation
            y_equalized = pseudo_inverse(y_symbols, H)
            X_Initial = BPSK_optimal(y_equalized, H, constellation, bound=0.5)
            demod_bits = Tx.demodulate(X_Initial, 'hard')
            # decoding and CRC
            decode_hard_bits = cc.viterbi_decode(demod_bits, trellis, tb_depth, decoding_type='hard')
            decode_bits = decode_hard_bits[ : message_bits.shape[0]]
            checksum = CRC_decode(decode_bits, remainder, CRC_key)

            if checksum > 0:
                xhat_indices = np.zeros(y_symbols.shape, dtype=int)
                y_tensor = torch.from_numpy(np.real(y_equalized)).float()
                for i in range(0, y_tensor.shape[0]):
                    xhat_tensor = model(torch.Tensor([y_tensor[i]]))
                    xhat_indices[i] = torch.argmax(xhat_tensor).item()
                xhat_symbols = Tx.constellation[xhat_indices]
                demod_bits = Tx.demodulate(xhat_symbols, 'hard')
                decode_hard_bits = cc.viterbi_decode(demod_bits, trellis, tb_depth, decoding_type='hard')
                decode_bits = decode_hard_bits[ : message_bits.shape[0]]
                checksum_new = CRC_decode(decode_bits, remainder, CRC_key)
                if checksum_new > 0:
                    fail_num += 1

            BER_decoded_vector[jj] = bit_err_rate(message_bits, decode_bits)
            BER_uncoded_vector[jj] = bit_err_rate(coded_bits, demod_bits)

        BER_uncoded_block[bb] = np.mean(BER_uncoded_vector)
        BER_decoded_block[bb] = np.mean(BER_decoded_vector)

    BER_uncoded_main[dd] = np.mean(BER_uncoded_block)
    BER_decoded_main[dd] = np.mean(BER_decoded_block)
    pass_rate[dd] = 1 - (fail_num/(block_size * num_blocks))

    print("--- dd=%d --- SNR = %.1f dB --- %s seconds ---" % (dd, SNRdB_range[dd], time.time() - start_time))
    print("uncoded BER: ")
    print(BER_uncoded_main)
    print("decoded BER: ")
    print(BER_decoded_main)
    print("CRC pass rate: ")
    print(pass_rate)


# display results """
plt.plot(SNRdB_range, BER_uncoded_main, '-ro', label='after demodulation')
plt.plot(SNRdB_range, BER_decoded_main, '-bo', label='after decoding')
plt.yscale('log')
plt.ylim(1e-6, 1)

plt.xlabel('SNR(dB)')
plt.ylabel('BER')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('./figures/BPSK_det3_Tx100_Rx100_FlatFading_bound5e-1.png')   