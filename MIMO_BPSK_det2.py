import numpy as np 
import torch
from torch import nn
from CommPy.utils import *
from CommPy.modulation import PSKModem
import CommPy.channelcoding.convcode as cc
from CommPy.CRC import CRC_encode, CRC_decode
from torch.utils.data import Dataset, DataLoader
from dataset import MIMO_BPSKDataset
from networks.FCNet import MIMO_BPSK_MLP
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os


""" Global Variables"""
N = 4                 # number of users
M = 4                 # number of receiving antennas
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
# storage_path = './filtered_data/MIMO_BPSK_Tx100_Rx100_AWGN_MMSE.npy'
# constellation symbols follow uniform distribution at first
# lld_k_Initial = np.log(1/mod * np.ones(mod)) 


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


def apply_network(y_symbols, Net, cuda=False):
    # don't use cuda now
    
    y_tensor = torch.from_numpy(np.real(y_symbols)).float()
    
    xhat_tensor = Net(y_tensor)
    xhat_dec = torch.argmax(xhat_tensor.detach()).item()
    xhat_indices = decimal2Qbase(xhat_dec, mod, N)

    return xhat_indices


def eval_once(storage_path, params, Net, modem, trellis, pass_rate, cuda=False):
    
    new_length = params['new_length']
    tb_depth = params['tb_depth']

    pass_num = num_packets * pass_rate
    SNR_interval = SNRdB_range[1] - SNRdB_range[0]

    data = np.load(storage_path, allow_pickle=True)
    SNR_list = data.item().get('SNRdB')
    message_array = data.item().get('message_bits')
    x_indices_array = data.item().get('x_indices')
    y_array = data.item().get('y_symbols')
    datalen = len(SNR_list)

    print('--------------Evaluation---------------')
    for i in tqdm(range(1, datalen)):
        # in each packet
        SNRdB = SNR_list[i]
        # [N, length]
        message_bits = message_array[i]
        remainder_list = []
        for k in range(0, N):
            remainder, codeword = CRC_encode(message_bits[k], CRC_key)
            remainder_list.append(remainder)
        # [N, new_length]
        y_symbols_array = y_array[i]

        xhat_indices_array = np.zeros([N, new_length], dtype=int)
        for l in range(0, new_length):
            xhat_indices_array[:, l] = apply_network(y_symbols_array[:, l], Net, cuda)
        
        pass_flag = 1
        for k in range(0, N):
            xhat_symbols = modem.constellation[xhat_indices_array[k]]
            demod_bits = modem.demodulate(xhat_symbols, 'hard')
            decode_hard_bits = cc.viterbi_decode(demod_bits, trellis, tb_depth, decoding_type='hard')
            decode_bits = decode_hard_bits[ : (length * bit_width)]
            checksum = CRC_decode(decode_bits, remainder_list[k], CRC_key)
            if checksum > 0:
                pass_flag = 0

        if pass_flag == 1:
            pass_num[int(SNRdB/SNR_interval)] += 1

    new_pass_rate = pass_num / num_packets

    return new_pass_rate


def train_eval(storage_path, pass_rate, ckpt_load_path, ckpt_save_dir, cuda):
    
    # create modem
    modem = PSKModem(mod)
    # create trellis
    generator_matrix = np.array([[5, 7]])
    constraint_len = 7
    trellis, tb_depth = get_trellis(mod, generator_matrix, constraint_len)
    # compute new_length: the number of symbols each packet after decoding
    new_length = get_encoded_length(trellis)

    params = {
        'new_length': new_length,
        'tb_depth': tb_depth
    }

    model = MIMO_BPSK_MLP(mod, N)
    if cuda:
        model = model.cuda()
    if ckpt_load_path is not None:
        model.load_state_dict(torch.load(ckpt_load_path))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    training_data = MIMO_BPSKDataset(storage_path, mod, N)
    dataloader = DataLoader(training_data, batch_size = 1)

    pr_record = np.expand_dims(pass_rate, axis=0)
    
    num_epochs = 2
    for t in range(0, num_epochs):
        running_loss = 0
        for batch, packet in tqdm(enumerate(dataloader)):
            y_tensor = packet['input']
            x_tensor = packet['label']
            if cuda:
                y_tensor = y_tensor.cuda()
                x_tensor = x_tensor.cuda()
            
            xhat_tensor = model(y_tensor)
            loss = loss_fn(xhat_tensor, x_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch+1)%5000 == 0:
                print(running_loss/5000)
                running_loss = 0

        if (t+1)%1 == 0:
            new_pass_rate = eval_once(storage_path, params, model, modem, trellis, pass_rate, cuda)
            print(new_pass_rate)
            pr_record = np.append(pr_record, np.expand_dims(new_pass_rate, axis=0), axis=0)
            ckpt_save_path = os.path.join(ckpt_save_dir, 'randomH_MMSEdet_epoch{:d}.pt'.format(t))
            torch.save(model.state_dict(), ckpt_save_path)

    return pr_record


def main():
    storage_path = './filtered_data/MIMO_BPSK_Tx4_Rx4_FixH_MMSEdet_CH9.npy'
    #storage_path = './filtered_data/test.npy'  
    pass_rate = np.array([0., 0.001, 0., 0.004, 0.004, 0.004, 0.004, 0.016, 0.033, 0.115, 0.3])   
    ckpt_load_path = None
    ckpt_save_dir = './checkpoints/MIMO_BPSK_Tx4_Rx4_Adam'
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)
    cuda = False
    pr_record = train_eval(storage_path, pass_rate, ckpt_load_path, ckpt_save_dir, cuda)
    print(pr_record)


if __name__ == '__main__':
    main()