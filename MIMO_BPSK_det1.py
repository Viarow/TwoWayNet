import numpy as np 
import time
from CommPy.utils import *
from CommPy.modulation import PSKModem
import CommPy.channelcoding.convcode as cc
from CommPy.CRC import CRC_encode, CRC_decode
from CommPy.detector import MMSE_BPSK, maximum_likelihood
from initialization import BPSK_optimal, MMSE_initial
from IterativeSIC import iterative_SIC
import matplotlib.pyplot as plt
from tqdm import tqdm


""" Global Variables"""
N = 4                 # number of users
M = 4                # number of receiving antennas
length = 100          # number of symbols in one packet
num_packets = 1000    # number of packets for each SNR
mod = 2               # modulation order, for BPSK is 2
bit_width = int(np.log2(mod))  # number of bits in one symbol
SNRdB_range = np.arange(0, 11, 1)
H_mean = 0            # channel matrix distribution
H_var = 1             # channel matrix distribution
rate = 1/2            # convolutional code rate
CRC_key = '100000111' # CRC-8 polynomial
# path to store packets that do not pass CRC
storage_path = './filtered_data/MIMO_BPSK_Tx4_Rx4_FixH_MMSEdet_CH9.npy'
# constellation symbols follow uniform distribution at first
lld_k_Initial = np.log(1/mod * np.ones(mod)) 


def get_trellis(mod, generator_matrix, constraint_len):

    trellis = cc.Trellis(np.array([mod]), generator_matrix) # Trellis structure
    num_delay = np.array([constraint_len - 1])  # number of delay elements
    tb_depth = 5*(num_delay.sum() + 1)  # traceback depth
    
    return trellis, tb_depth


def get_encoded_length(trellis):

    tx_sample = np.random.randint(0, 2, (bit_width * length, ))
    tx_coded = cc.conv_encode(tx_sample, trellis)
    new_length = int(tx_coded.shape[0] / bit_width)

    return new_length


def get_storage_dict(new_length):
    void_x = np.zeros([N, new_length]) + 1j * np.zeros([N, new_length])
    void_y = np.zeros([N, new_length]) + 1j * np.zeros([N, new_length])
    storage_dict = {
        'SNRdB': [0],
        'message_bits': np.expand_dims(np.zeros([N, length * bit_width], dtype=int), axis=0),
        'coded_bits': np.expand_dims(np.zeros([N, new_length * bit_width], dtype=int), axis=0),
        'x_indices': np.expand_dims(np.zeros([N, new_length], dtype=int), axis=0),
        'x_symbols': np.expand_dims(void_x, axis=0),
        'y_symbols': np.expand_dims(void_y, axis=0)
    }

    return storage_dict


""" TODO: Main Change Here """
def single_time_transmit(x_symbols, signal_power, var_noise, H, constellation):
    # compute noise
    noise_real = 0. + np.sqrt(var_noise)*np.random.randn(M)
    #noise_imag = 0. + np.sqrt(0.5 * var_noise)*np.random.randn(M)
    noise_imag = np.zeros(noise_real.shape)

    # channel: AWGN
    # y_symbols = x_symbols + noise
    y_symbols = np.matmul(H, x_symbols) + (noise_real + 1j * noise_imag)

    """equalization"""
    y_equalized = y_symbols
    # lamda = var_noise/signal_power
    # y_equalized = MMSE_BPSK(H, y_symbols, lamda, constellation)

    """ detection """
    lamda = var_noise/signal_power
    xhat_symbols = MMSE_BPSK(H, y_symbols, lamda, constellation)
    # num_iter = 5
    # xhat_mat, prob = iterative_SIC(y_equalized, var_equalized, var_noise, y_symbols, H, constellation, num_iter)
    # xhat_indices = xhat_mat[num_iter-1]
    # xhat_symbols = constellation[xhat_indices]
    # xhat_indices, xhat_symbols = maximum_likelihood(y_symbols, H, mod, constellation, N)
    
    return y_equalized, xhat_symbols


def main():

    # create modem
    modem = PSKModem(mod)
    constellation = modem.constellation

    # create trellis
    generator_matrix = np.array([[5, 7]])
    constraint_len = 7
    trellis, tb_depth = get_trellis(mod, generator_matrix, constraint_len)

    # compute new_length: the number of symbols each packet after decoding
    new_length = get_encoded_length(trellis)

    # init a dictionary to store failed packets
    storage_dict = get_storage_dict(new_length)

    # compute signal power
    signal_power = np.abs(constellation[0])

    # record BER after decoding and CRC pass rate
    SER_uncoded_main = np.zeros([N, SNRdB_range.shape[0]])
    BER_decoded_main = np.zeros([N, SNRdB_range.shape[0]])
    pass_rate = np.zeros(SNRdB_range.shape) 

    #H_real = H_mean + np.sqrt(0.5 * H_var)*np.random.randn(M, N)
    #H_imag = H_mean + np.sqrt(0.5 * H_var)*np.random.randn(M, N)
    # condition number 9.74
    # H_real = np.eye(M)
    H_real = np.array([[-1.02865866,  1.33491489, -0.49420349, -1.85878384],
        [ 0.67367862, -0.71196127, -0.69004872, -0.88224427],
        [-1.15128467,  0.03224837, -0.84383447, -0.65113325],
        [-1.18420638, -0.58942956, -0.35515315, -0.14245944]])
    H_imag = np.zeros(H_real.shape)
    H = H_real + 1j * H_imag

    # main loops
    for dd in range(0, SNRdB_range.shape[0]):
        start_time = time.time()

        var_noise = signal_power * H_var * np.power(10, -0.1*SNRdB_range[dd])
        SER_uncoded_block = np.zeros([N, num_packets])
        BER_decoded_block = np.zeros([N, num_packets])
        fail_num = 0

        for bb in tqdm(range(0, num_packets)):
            # random H
            # H_real = H_mean + np.sqrt(H_var)*np.random.randn(M, N)
            # H_imag = H_mean + np.sqrt(0.5 * H_var)*np.random.randn(M, N)
            # H_imag = np.zeros(H_real.shape)
            # H = H_real + 1j * H_imag

            # data bits in one packet
            message_bits = np.random.randint(0, 2, (N, length * bit_width))
            # create CRC remainder list
            remainder_list = []
            # initialize code bits array
            coded_bits = np.random.randint(0, 2, (N, new_length * bit_width))
            # initialize x indices array
            x_indices_array = np.zeros([N, new_length], dtype=int)
            # intialize x symbols array
            x_symbols_array = np.zeros([N, new_length]) + 1j * np.zeros([N, new_length])

            for kk in range(0, N):
                # get CRC remiander for each user
                remainder, codeword = CRC_encode(message_bits[kk], CRC_key)
                remainder_list.append(remainder)
                # encode the bit sequence for each user
                coded_bits[kk] = cc.conv_encode(message_bits[kk], trellis)
                # modulate the encoded bits for each user
                indices_k, symbols_k = modem.modulate(coded_bits[kk])
                x_indices_array[kk] = indices_k
                x_symbols_array[kk] = symbols_k

            # transmit through channel, equalize, and detect
            xhat_indices_array = np.zeros([N, new_length], dtype=int)
            xhat_symbols_array = np.zeros([N, new_length]) + 1j * np.zeros([N, new_length])
            y_equalized_array = np.zeros([N, new_length]) + 1j * np.zeros([N, new_length])  
            for ll in range(0, new_length):
                y_equalized, xhat_symbols = single_time_transmit(x_symbols_array[:, ll], signal_power, var_noise, H, constellation)
                # xhat_indices_array[:, ll] = xhat['indices']
                # xhat_symbols_array[:, ll] = xhat['symbols']
                xhat_symbols_array[:, ll] = xhat_symbols
                y_equalized_array[:, ll] = y_equalized

            # demodulate, decode and CRC, if one user fails, store the info for all users
            pass_flag = 1
            for kk in range(0, N):

                demod_bits = modem.demodulate(xhat_symbols_array[kk], 'hard')
                xhat_indices, _ = modem.modulate(demod_bits)
                SER_uncoded_block[kk, bb] = bit_err_rate(x_indices_array[kk], xhat_indices)

                decode_hard_bits = cc.viterbi_decode(demod_bits, trellis, tb_depth, decoding_type='hard')
                decode_bits = decode_hard_bits[ : (length * bit_width)]
                BER_decoded_block[kk, bb] = bit_err_rate(message_bits[kk], decode_bits)

                checksum = CRC_decode(decode_bits, remainder_list[kk], CRC_key)
                # if any user fails
                if checksum > 0:
                    pass_flag = 0
            
            if pass_flag == 0:
                fail_num += 1
                storage_dict['SNRdB'].append(SNRdB_range[dd])
                storage_dict['message_bits'] = np.append(storage_dict['message_bits'], np.expand_dims(message_bits, axis=0), axis=0)
                storage_dict['coded_bits'] = np.append(storage_dict['coded_bits'], np.expand_dims(coded_bits, axis=0), axis=0)
                storage_dict['x_indices'] = np.append(storage_dict['x_indices'], np.expand_dims(x_indices_array, axis=0), axis=0)
                storage_dict['x_symbols'] = np.append(storage_dict['x_symbols'], np.expand_dims(x_symbols_array, axis=0), axis=0)
                storage_dict['y_symbols'] = np.append(storage_dict['y_symbols'], np.expand_dims(y_equalized_array, axis=0), axis=0)
            
            # iteration over all packets at one SNR level ends here
        
        BER_decoded_main[:, dd] = np.mean(BER_decoded_block, axis=1)
        SER_uncoded_main[:, dd] = np.mean(SER_uncoded_block, axis=1)
        pass_rate[dd] = 1 - (fail_num/num_packets)

        print("--- dd=%d --- SNR = %.1f dB --- %s seconds ---" % (dd, SNRdB_range[dd], time.time() - start_time))
        print('detector SER')
        print(SER_uncoded_main)
        print("decoded BER: ")
        print(BER_decoded_main)
        print("CRC pass rate: ")
        print(pass_rate)

    # iteration over all SNR levels ends here
    np.save(storage_path, storage_dict)


if __name__ == '__main__':
    main()