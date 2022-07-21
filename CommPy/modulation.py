# Authors: CommPy contributors
# License: BSD 3-Clause

"""
==================================================
Modulation Demodulation (:mod:`commpy.modulation`)
==================================================
.. autosummary::
   :toctree: generated/
   PSKModem             -- Phase Shift Keying (PSK) Modem.
   QAMModem             -- Quadrature Amplitude Modulation (QAM) Modem.
   ofdm_tx              -- OFDM Transmit Signal Generation
   ofdm_rx              -- OFDM Receive Signal Processing
   mimo_ml              -- MIMO Maximum Likelihood (ML) Detection.
"""

from numpy import arange, array, zeros, pi, sqrt, log2, argmin, \
    hstack, repeat, tile, dot, shape, concatenate, exp, \
    log, vectorize, empty, eye, kron, inf, full, abs, newaxis, minimum, clip, fromiter
from numpy.fft import fft, ifft
from numpy.linalg import qr, norm
from sympy.combinatorics.graycode import GrayCode
from CommPy.utils import bitarray2dec, dec2bitarray, signal_power

__all__ = ['PSKModem', 'QAMModem', 'ofdm_tx', 'ofdm_rx', 'mimo_ml']


class Modem:

    """ Creates a custom Modem object.
        Parameters
        ----------
        constellation : array-like with a length which is a power of 2
                        Constellation of the custom modem
        Attributes
        ----------
        constellation : 1D-ndarray of complex
                        Modem constellation. If changed, the length of the new constellation must be a power of 2.
        Es            : float
                        Average energy per symbols.
        m             : integer
                        Constellation length.
        num_bits_symb : integer
                        Number of bits per symbol.
        Raises
        ------
        ValueError
                        If the constellation is changed to an array-like with length that is not a power of 2.
        """
    
    def __init__(self, constellation, reorder_as_gray=True):
        """ Creates a custom Modem object. """

        if reorder_as_gray:
            m = log2(len(constellation))
            gray_code_sequence = GrayCode(m).generate_gray()
            gray_code_sequence_array = fromiter((int(g, 2) for g in gray_code_sequence), int, len(constellation))
            self.constellation = array(constellation)[gray_code_sequence_array.argsort()]
        else:
            self.constellation = constellation


    def modulate(self, input_bits):
        """ Modulate (map) an array of bits to constellation symbols.
        Parameters
        ----------
        input_bits : 1D ndarray of ints
            Inputs bits to be modulated (mapped).
        Returns
        -------
        baseband_symbols : 1D ndarray of complex floats
            Modulated complex symbols.
        """
        mapfunc = vectorize(lambda i:
                            self._constellation[bitarray2dec(input_bits[i:i + self.num_bits_symbol])])

        baseband_symbols = mapfunc(arange(0, len(input_bits), self.num_bits_symbol))

        indexfunc = vectorize(lambda i: bitarray2dec(input_bits[i:i + self.num_bits_symbol]))
        baseband_indices = indexfunc(arange(0, len(input_bits), self.num_bits_symbol))

        return baseband_indices, baseband_symbols


    def demodulate(self, input_symbols, demod_type, noise_var=0):
        """ Demodulate (map) a set of constellation symbols to corresponding bits.
        Parameters
        ----------
        input_symbols : 1D ndarray of complex floats
            Input symbols to be demodulated.
        demod_type : string
            'hard' for hard decision output (bits)
            'soft' for soft decision output (LLRs)
        noise_var : float
            AWGN variance. Needs to be specified only if demod_type is 'soft'
        Returns
        -------
        demod_bits : 1D ndarray of ints
            Corresponding demodulated bits.
        """
        if demod_type == 'hard':
            index_list = abs(input_symbols - self._constellation[:, None]).argmin(0)
            demod_bits = dec2bitarray(index_list, self.num_bits_symbol)

        elif demod_type == 'soft':
            demod_bits = zeros(len(input_symbols) * self.num_bits_symbol)
            for i in arange(len(input_symbols)):
                current_symbol = input_symbols[i]
                for bit_index in arange(self.num_bits_symbol):
                    llr_num = 0
                    llr_den = 0
                    for bit_value, symbol in enumerate(self._constellation):
                        if (bit_value >> bit_index) & 1:
                            llr_num += exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                        else:
                            llr_den += exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                    demod_bits[i * self.num_bits_symbol + self.num_bits_symbol - 1 - bit_index] = log(llr_num / llr_den)
        else:
            raise ValueError('demod_type must be "hard" or "soft"')

        return demod_bits

    def plot_constellation(self):
        """ Plot the constellation """
        plt.scatter(self.constellation.real, self.constellation.imag)

        for symb in self.constellation:
            plt.text(symb.real + .2, symb.imag, self.demodulate(symb, 'hard'))

        plt.title('Constellation')
        plt.grid()
        plt.show()

    @property
    def constellation(self):
        """ Constellation of the modem. """
        return self._constellation

    @constellation.setter
    def constellation(self, value):
        # Check value input
        num_bits_symbol = log2(len(value))
        if num_bits_symbol != int(num_bits_symbol):
            raise ValueError('Constellation length must be a power of 2.')

        # Set constellation as an array
        self._constellation = array(value)

        # Update other attributes
        self.Es = signal_power(self.constellation)
        self.m = self._constellation.size
        self.num_bits_symbol = int(num_bits_symbol)


class PSKModem(Modem):

    """ Creates a Phase Shift Keying (PSK) Modem object.
        Parameters
        ----------
        m : int
            Size of the PSK constellation.
        Attributes
        ----------
        constellation : 1D-ndarray of complex
                        Modem constellation. If changed, the length of the new constellation must be a power of 2.
        Es            : float
                        Average energy per symbols.
        m             : integer
                        Constellation length.
        num_bits_symb : integer
                        Number of bits per symbol.
        Raises
        ------
        ValueError
                        If the constellation is changed to an array-like with length that is not a power of 2.
    """
    def __init__(self, m):
        """ Creates a Phase Shift Keying (PSK) Modem object. """

        num_bits_symbol = log2(m)
        if num_bits_symbol != int(num_bits_symbol):
            raise ValueError('Constellation length must be a power of 2.')

        super().__init__(exp(1j * arange(0, 2 * pi, 2 * pi / m)))


class QAMModem(Modem):
    """ Creates a Quadrature Amplitude Modulation (QAM) Modem object.
        Parameters
        ----------
        m : int
            Size of the PSK constellation.
        Attributes
        ----------
        constellation : 1D-ndarray of complex
                        Modem constellation. If changed, the length of the new constellation must be a power of 2.
        Es            : float
                        Average energy per symbols.
        m             : integer
                        Constellation length.
        num_bits_symb : integer
                        Number of bits per symbol.
        Raises
        ------
        ValueError
                        If the constellation is changed to an array-like with length that is not a power of 2.
                        If the parameter m would lead to an non-square QAM during initialization.
    """
    def __init__(self, m):
        """ Creates a Quadrature Amplitude Modulation (QAM) Modem object.
        Parameters
        ----------
        m : int
            Size of the QAM constellation. Must lead to a square QAM (ie sqrt(m) is an integer).
        Raises
        ------
        ValueError
                        If m would lead to an non-square QAM.
        """

        num_symb_pam = sqrt(m)
        if num_symb_pam != int(num_symb_pam):
            raise ValueError('m must lead to a square QAM.')

        pam = arange(-num_symb_pam + 1, num_symb_pam, 2)
        constellation = tile(hstack((pam, pam[::-1])), int(num_symb_pam) // 2) * 1j + pam.repeat(num_symb_pam)
        super().__init__(constellation)


def ofdm_tx(x, nfft, nsc, cp_length):
    """ OFDM Transmit signal generation """

    nfft = float(nfft)
    nsc = float(nsc)
    cp_length = float(cp_length)
    ofdm_tx_signal = array([])

    for i in range(0, shape(x)[1]):
        symbols = x[:, i]
        ofdm_sym_freq = zeros(nfft, dtype=complex)
        ofdm_sym_freq[1:(nsc / 2) + 1] = symbols[nsc / 2:]
        ofdm_sym_freq[-(nsc / 2):] = symbols[0:nsc / 2]
        ofdm_sym_time = ifft(ofdm_sym_freq)
        cp = ofdm_sym_time[-cp_length:]
        ofdm_tx_signal = concatenate((ofdm_tx_signal, cp, ofdm_sym_time))

    return ofdm_tx_signal


def ofdm_rx(y, nfft, nsc, cp_length):
    """ OFDM Receive Signal Processing """

    num_ofdm_symbols = int(len(y) / (nfft + cp_length))
    x_hat = zeros([nsc, num_ofdm_symbols], dtype=complex)

    for i in range(0, num_ofdm_symbols):
        ofdm_symbol = y[i * nfft + (i + 1) * cp_length:(i + 1) * (nfft + cp_length)]
        symbols_freq = fft(ofdm_symbol)
        x_hat[:, i] = concatenate((symbols_freq[-nsc / 2:], symbols_freq[1:(nsc / 2) + 1]))

    return x_hat


def mimo_ml(y, h, constellation):
    """ MIMO ML Detection.
    parameters
    ----------
    y : 1D ndarray of complex floats
        Received complex symbols (shape: num_receive_antennas x 1)
    h : 2D ndarray of complex floats
        Channel Matrix (shape: num_receive_antennas x num_transmit_antennas)
    constellation : 1D ndarray of complex floats
        Constellation used to modulate the symbols
    """
    _, n = h.shape
    m = len(constellation)
    x_ideal = empty((n, pow(m, n)), complex)
    for i in range(0, n):
        x_ideal[i] = repeat(tile(constellation, pow(m, i)), pow(m, n - i - 1))
    min_idx = argmin(norm(y[:, None] - dot(h, x_ideal), axis=0))
    x_r = x_ideal[:, min_idx]

    return x_r