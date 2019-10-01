from __future__ import absolute_import

import numpy as np
import os
import gzip
import struct
import functools
import array
import operator

class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass

def parse_idx(fd):

    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items, len(data)))

    return np.array(data).reshape(dimension_sizes)

def test_images():
    cur_dir = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    with gzip.open(os.path.join(cur_dir,"t10k-images-idx3-ubyte.gz"), 'rb') as fd:
        return parse_idx(fd)       

def test_labels():
    cur_dir = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    with gzip.open(os.path.join(cur_dir,"t10k-labels-idx1-ubyte.gz"), 'rb') as fd:
        return parse_idx(fd)       
