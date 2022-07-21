import numpy as np 
import torch
from torch import nn
from CommPy.utils import *
from CommPy.modulation import QAMModem, PSKModem
import CommPy.channelcoding.convcode as cc
from CommPy.CRC import CRC_encode, CRC_decode
from torch.utils.data import Dataset, DataLoader
from dataset import BPSK_Symbol2Bits
from networks.FCNet import BinaryClassifier, FlatFadingClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


def apply_network(y_tensor, Net, cuda):
    
    xhat_bits = np.zeros(y_tensor.shape, dtype=int)
    if cuda:
        y_tensor = y_tensor.cuda()
        for i in range(0, y_tensor.shape[0]):
            xhat_tensor = Net(torch.Tensor([y_tensor[i]])).cpu().data()
            xhat_bits[i] = torch.argmax(xhat_tensor.detach()).item()
    else:
        for i in range(0, y_tensor.shape[0]):
            xhat_tensor = Net(torch.Tensor([y_tensor[i]]))
            xhat_bits[i] = torch.argmax(xhat_tensor.detach()).item()

    return xhat_bits


def eval_once(storage_path, params, Net, modem, trellis, pass_rate, cuda=False):
    # params{m, n, mod, H_width, H_height, CRC_key, num_packets, tb_depth, SNRdB_range}
    pass_num = params['num_packets'] * pass_rate
    SNR_interval = params['SNRdB_range'][1] - params['SNRdB_range'][0]

    data = np.load(storage_path, allow_pickle=True)
    SNR_list = data.item().get('SNRdB')
    message_array = data.item().get('message_bits')
    coded_bits_array = data.item().get('coded_bits')
    y_array = data.item().get('y_symbols')
    datalen = len(SNR_list)

    print('--------------Evaluation---------------')
    for k in tqdm(range(1, datalen)):
        SNRdB = SNR_list[k]
        message_bits = message_array[k]
        coded_bits = coded_bits_array[k]
        remainder, codeword = CRC_encode(message_bits, params['CRC_key'])
        y_symbols = y_array[k]
        y_tensor = torch.from_numpy(np.real(y_symbols)).float()

        xhat_bits = apply_network(y_tensor, Net, cuda)
        xhat_symbols = np.zeros(y_symbols.shape) + 1j * np.zeros(y_symbols.shape)
        for i in range(0, xhat_bits.shape[0]):
            if xhat_bits[i] == 0:
                xhat_symbols[i] = modem.constellation[0]
            else:
                xhat_symbols[i] = modem.constellation[1]            

        demod_bits = modem.demodulate(xhat_symbols, 'hard')
        decode_hard_bits = cc.viterbi_decode(demod_bits, trellis, params['tb_depth'], decoding_type='hard')
        decode_bits = decode_hard_bits[ : message_bits.shape[0]]
        checksum = CRC_decode(decode_bits, remainder, params['CRC_key'])
        if checksum == 0:
            pass_num[int(SNRdB/SNR_interval)] += 1
        
    new_pass_rate = pass_num / params['num_packets']
    
    return new_pass_rate


def train_eval(storage_path, pass_rate, ckpt_load_path, ckpt_save_path, cuda):

    params = {
        'm': 100,
        'n': 100,
        'mod': 2,
        'H_width': 0,
        'H_height': 0,
        'CRC_key': '100000111',
        'num_packets': 1000,
        'tb_depth':0,
        'SNRdB_range': np.arange(0, 18, 2),
        'learning_rate': 1e-3,
        'num_epochs': 1,
        'batch_size': 1
    }

    Tx = PSKModem(params['mod'])
    constellation = Tx.constellation
    generator_matrix = np.array([[5, 7]])
    trellis = cc.Trellis(np.array([params['mod']]), generator_matrix)
    rate = 1/2
    constraint_len = 7 
    num_delay = np.array([constraint_len - 1])
    params['tb_depth'] = 5*(num_delay.sum() + 1)

    bit_width = int(np.log2(params['mod']))
    tx_sample = np.random.randint(0, 2, (bit_width*params['n'], ))
    tx_coded = cc.conv_encode(tx_sample, trellis)
    H_width = int(tx_coded.shape[0] / bit_width)
    params['H_width'] = H_width
    rx_sample = np.random.randint(0, 2, (bit_width*params['m'], ))
    rx_coded = cc.conv_encode(rx_sample, trellis)
    H_height = int(rx_coded.shape[0] / bit_width)
    params['H_height'] = H_height

    model = BinaryClassifier()
    if cuda:
        model = model.cuda()
    if ckpt_load_path is not None:
        model.load_state_dict(torch.load(ckpt_load_path))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])
    training_data = BPSK_Symbol2Bits(storage_path)
    dataloader = DataLoader(training_data, batch_size = params['batch_size'])

    pr_record = np.expand_dims(pass_rate, axis=0)
    
    for t in range(0, params['num_epochs']):
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

        if (t+1)%1 == 0:
            new_pass_rate = eval_once(storage_path, params, model, Tx, trellis, pass_rate, cuda)
            print(new_pass_rate)
            torch.save(model.state_dict(), ckpt_save_path)
            pr_record = np.append(pr_record, np.expand_dims(new_pass_rate, axis=0), axis=0)

    return pr_record

def main():
    storage_path = './filtered_data/BPSK_Tx100_Rx100_FlatFading_bound5e-1.npy'   
    #storage_path = './filtered_data/test.npy'  
    pass_rate = np.array([0.111, 0.243, 0.334, 0.513, 0.694, 0.734, 0.822, 0.872, 0.94])   
    ckpt_load_path = None
    ckpt_save_path = './checkpoints/BPSK_Tx100_Rx100_FlatFading_bound5e-1/epoch_1.pt'
    cuda = False
    pr_record = train_eval(storage_path, pass_rate, ckpt_load_path, ckpt_save_path, cuda)
    print(pr_record)


if __name__ == '__main__':
    main()