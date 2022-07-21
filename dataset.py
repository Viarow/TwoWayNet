import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from CommPy.modulation import PSKModem
from CommPy.utils import Qbase2dec


class BPSK_FilteredDataset(Dataset):

    def __init__(self, storage_path, transform=None):
        data = np.load(storage_path, allow_pickle=True)
        self.x_array = data.item().get('x_symbols')
        self.y_array = data.item().get('y_symbols')

    def __len__(self):
        return self.x_array.shape[0]

    def __getitem__(self, idx):

        x_symbols = self.x_array[idx]
        x_real = torch.from_numpy(np.real(x_symbols)).float()
        x_imag = torch.from_numpy(np.imag(x_symbols)).float()
        x_tensor = torch.cat((x_real, x_imag))

        y_symbols = self.y_array[idx]
        y_real = torch.from_numpy(np.real(y_symbols)).float()
        y_imag = torch.from_numpy(np.imag(y_symbols)).float()
        y_tensor = torch.cat((y_real, y_imag))

        packet = {'input':y_real, 'label':x_real}
        return packet


class BPSK_SymbolDataset(Dataset):

    def __init__(self, storage_path, transform=None):
        data = np.load(storage_path, allow_pickle=True)
        x_array = data.item().get('x_symbols')
        #coded_bits_array = data.item().get('coded_bits')
        y_array = data.item().get('y_symbols')
        self.x_list = np.reshape(x_array, (x_array.shape[0] * x_array.shape[1], ))
        #self.coded_bits_list = np.reshape(coded_bits_array, (coded_bits_array[0] * coded_bits_array[1]))
        self.y_list = np.reshape(y_array, (y_array.shape[0] * y_array.shape[1], ))

    def __len__(self):
        return self.x_list.shape[0]

    def __getitem__(self, idx):
        
        x_symbol = np.array([self.x_list[idx]])
        x_real = torch.from_numpy(np.real(x_symbol)).float()
        #label = self.coded_bits_list[idx]
        y_symbol = np.array([self.y_list[idx]])
        y_real = torch.from_numpy(np.real(y_symbol)).float()
        packet = {'input': y_real, 'label': x_real}

        return packet


class BPSK_Symbol2Bits(Dataset):

    def __init__(self, storage_path, transform=None):
        data = np.load(storage_path, allow_pickle=True)
        #x_array = data.item().get('x_symbols')
        coded_bits_array = data.item().get('coded_bits')
        y_array = data.item().get('y_symbols')
        #self.x_list = np.reshape(x_array, (x_array.shape[0] * x_array.shape[1]))
        self.coded_bits_list = np.reshape(coded_bits_array, (coded_bits_array.shape[0] * coded_bits_array.shape[1], ))
        self.y_list = np.reshape(y_array, (y_array.shape[0] * y_array.shape[1], ))

    def __len__(self):
        return self.y_list.shape[0]

    def __getitem__(self, idx):
        
        #x_symbol = np.array([self.x_list[idx]])
        #x_real = torch.from_numpy(np.real(x_symbol)).float()
        label = self.coded_bits_list[idx]
        y_symbol = np.array([self.y_list[idx]])
        y_real = torch.from_numpy(np.real(y_symbol)).float()
        packet = {'input': y_real, 'label': label}

        return packet


class QAM4_Symbol2Bits(Dataset):

    def __init__(self, storage_path, transform=None):
        data = np.load(storage_path, allow_pickle=True)
        x_indices_array = data.item().get('x_indices')
        y_array = data.item().get('y_symbols')
        self.x_indices_list = np.reshape(x_indices_array, (x_indices_array.shape[0]*x_indices_array.shape[1], ))
        self.y_list = np.reshape(y_array, (y_array.shape[0] * y_array.shape[1], ))
    
    def __len__(self):
        return self.y_list.shape[0]

    def __getitem__(self, idx):
        
        label = self.x_indices_list[idx]
        y_symbol = self.y_list[idx]
        y_cat = np.array([np.real(y_symbol), np.imag(y_symbol)])
        y_tensor = torch.from_numpy(y_cat).float()
        packet = {'input': y_tensor, 'label':label}

        return packet


class MIMO_BPSKDataset(Dataset):

    def __init__(self, storage_path, mod, N, transform=None):

        self.mod = mod
        data = np.load(storage_path, allow_pickle=True)
        x_indices_array = data.item().get('x_indices')
        y_array = data.item().get('y_symbols')
        
        # N = x_indices_array.shape[1]
        self.N = N
        x_indices_array = np.transpose(x_indices_array, (0, 2, 1))
        self.x_indices_list = np.reshape(x_indices_array, (-1, N))

        y_array = np.transpose(y_array, (0, 2, 1))
        self.y_list = np.reshape(y_array, (-1, N))

    def __len__(self):
        return self.y_list.shape[0]

    def __getitem__(self, idx):

        label_cff = self.x_indices_list[idx]
        label = Qbase2dec(self.mod, label_cff)
        
        y_symbols = self.y_list[idx]
        y_real = torch.from_numpy(np.real(y_symbols)).float()

        packet = {'input': y_real, 'label':label}
        return packet        