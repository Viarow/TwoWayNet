import numpy as np

__all__ = ['array2string', 'xor', 'mod2div', 'CRC_encode', 'CRC_decode']


def array2string(x):
    # convert an 1-d binary numpy array to python string
    y = []
    num_bits = x.shape[0]
    for i in range(0, num_bits):
        if x[i] == 0:
            y.append('0')
        else:
            y.append('1')
            
    return ''.join(y)


def xor(a, b):
 
    # initialize result
    result = []
 
    # Traverse all bits, if bits are
    # same, then XOR is 0, else 1
    for i in range(1, len(b)):
        if a[i] == b[i]:
            result.append('0')
        else:
            result.append('1')
 
    return ''.join(result)
 
 
# Performs Modulo-2 division
def mod2div(divident, divisor):
 
    # Number of bits to be XORed at a time.
    pick = len(divisor)
 
    # Slicing the divident to appropriate
    # length for particular step
    tmp = divident[0 : pick]
 
    while pick < len(divident):
 
        if tmp[0] == '1':
 
            # replace the divident by the result
            # of XOR and pull 1 bit down
            tmp = xor(divisor, tmp) + divident[pick]
 
        else:   # If leftmost bit is '0'
            # If the leftmost bit of the dividend (or the
            # part used in each step) is 0, the step cannot
            # use the regular divisor; we need to use an
            # all-0s divisor.
            tmp = xor('0'*pick, tmp) + divident[pick]
 
        # increment pick to move further
        pick += 1
 
    # For the last n bits, we have to carry it out
    # normally as increased value of pick will cause
    # Index Out of Bounds.
    if tmp[0] == '1':
        tmp = xor(divisor, tmp)
    else:
        tmp = xor('0'*pick, tmp)
 
    checkword = tmp
    return checkword
 
# Function used at the sender side to encode
# data by appending remainder of modular division
# at the end of data.
def encodeData(data, key):
 
    l_key = len(key)
 
    # Appends n-1 zeroes at end of data
    appended_data = data + '0'*(l_key-1)
    remainder = mod2div(appended_data, key)
 
    # Append remainder in the original data
    codeword = data + remainder
    print("Remainder : ", remainder)
    print("Encoded Data (Data + Remainder) : ",
          codeword)


def CRC_encode(bitstream, key):
    # bitstream: 1-d int binary numpy array at the Tx
    # key: python string, CRC polynomial

    l_key = len(key)
    data = array2string(bitstream)

    appended_data = data + '0'*(l_key-1)
    remainder = mod2div(appended_data, key)
    codeword = data + remainder

    return remainder, codeword


def CRC_decode(bitstream, remainder, key):
    # bitstream: 1-d int binary numpy array at the Rx
    # remainder: python string to do error checking

    data = array2string(bitstream)
    appended_data = data + remainder
    result = mod2div(appended_data, key)

    checksum = 0
    for i in range(0, len(result)):
        if result[i] == '1':
            checksum += 1

    return checksum
