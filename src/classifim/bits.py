"""
Bit manipulation functions operating on np.ndarrays.
"""
import numpy as np
import numba

def countbits16(zs):
    """
    Returns the number of bits set in each element of the array zs.

    Args:
        zs : Array of 16-bit integers.

    Returns:
        Array of integers between 0 and 16 (inclusive): the bit counts.
    """
    res = (0x5555 & zs) + (0x5555 & (zs >> 1))
    res = (0x3333 & res) + (0x3333 & (res >> 2))
    res = (0x0f0f & res) + (0x0f0f & (res >> 4))
    res = (0x00ff & res) + (0x00ff & (res >> 8))
    return res

def countbits32_slow(zs):
    """
    Returns the number of bits set in each element of the array zs.

    DEPRECATED: Use countbits32 instead.

    Args:
        zs : Array of 32-bit integers.

    Returns:
        Array of integers between 0 and 32 (inclusive): the bit counts.
    """
    res = (0x55555555 & zs) + (0x55555555 & (zs >> 1))
    res = (0x33333333 & res) + (0x33333333 & (res >> 2))
    res = (0x0f0f0f0f & res) + (0x0f0f0f0f & (res >> 4))
    res = (0x00ff00ff & res) + (0x00ff00ff & (res >> 8))
    res = (0x0000ffff & res) + (0x0000ffff & (res >> 16))
    return res

def countbits32(zs):
    """
    Returns the number of bits set in each element of the array zs.

    Source:
    https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

    Args:
        zs : Array of 32-bit integers.

    Returns:
        Array of integers between 0 and 32 (inclusive): the bit counts.
    """
    assert zs.dtype == np.uint32
    zs = zs - ((zs >> 1) & 0x55555555)
    zs = (zs & 0x33333333) + ((zs >> 2) & 0x33333333)
    return ((zs + (zs >> 4) & 0xF0F0F0F) * 0x1010101) >> 24

def countbits64(zs):
    """
    Returns the number of bits set in each element of the array zs.

    Source:
    https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

    Args:
        zs : Array of 64-bit integers (dtype=np.uint64).

    Returns:
        Array of integers between 0 and 64 (inclusive): the bit counts.
    """
    assert zs.dtype == np.uint64
    zs = zs - ((zs >> 1) & 0x5555555555555555)
    zs = (zs & 0x3333333333333333) + ((zs >> 2) & 0x3333333333333333)
    zs = (zs + (zs >> 4)) & 0xf0f0f0f0f0f0f0f
    return (zs * 0x101010101010101) >> 56

@numba.njit(numba.uint64(numba.uint64))
def popcount64(v):
    # https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    v = v - ((v >> numba.uint64(1)) & numba.uint64(0x5555555555555555))
    v = (v & numba.uint64(0x3333333333333333)) + (
            (v >> numba.uint64(2)) & numba.uint64(0x3333333333333333))
    v = (v + (v >> numba.uint64(4))) & numba.uint64(0xf0f0f0f0f0f0f0f)
    v = numba.uint64(v * numba.uint64(0x101010101010101)) >> numba.uint64(56)
    return v

def reverse_bit_pairs_uint32(a):
    """
    Reverses bit pairs in each element of an uint32 np.ndarray.
    """
    assert a.dtype == np.uint32
    a = ((a & 0xffff0000) >> 16) | ((a & 0x0000ffff) << 16)
    a = ((a & 0xff00ff00) >> 8) | ((a & 0x00ff00ff) << 8)
    a = ((a & 0xf0f0f0f0) >> 4) | ((a & 0x0f0f0f0f) << 4)
    a = ((a & 0xcccccccc) >> 2) | ((a & 0x33333333) << 2)
    return a

def reverse_bits_uint32(a):
    """
    Reverses bits in each element of an uint32 np.ndarray.
    """
    assert a.dtype == np.uint32
    a = ((a & 0xffff0000) >> 16) | ((a & 0x0000ffff) << 16)
    a = ((a & 0xff00ff00) >> 8) | ((a & 0x00ff00ff) << 8)
    a = ((a & 0xf0f0f0f0) >> 4) | ((a & 0x0f0f0f0f) << 4)
    a = ((a & 0xcccccccc) >> 2) | ((a & 0x33333333) << 2)
    a = ((a & 0xaaaaaaaa) >> 1) | ((a & 0x55555555) << 1)
    return a

def extract_every_second_bit_uint32(a):
    """
    Extracts every second bit in each element of an uint32 np.ndarray.

    I.e. bits 0, 2, ... are extracted. E.g. 0x5 -> 0x3, 0x6 -> 0x2.
    """
    assert a.dtype == np.uint32
    # Comments show how a = 0xffffffff is transformed.
    a = (a & 0x11111111) | ((a & 0x44444444) >> 1) # 0x33333333
    a = (a & 0x0f0f0f0f) | ((a & 0xf0f0f0f0) >> 2) # 0x0f0f0f0f
    a = (a & 0x00ff00ff) | ((a & 0xff00ff00) >> 4) # 0x00ff00ff
    a = (a & 0x0000ffff) | ((a & 0xffff0000) >> 8) # 0x0000ffff
    return a

def spread_to_every_second_bit_uint32(a):
    """
    Spreads last 16 bits into bits 0, 2, ... for each element of a.
    """
    assert a.dtype == np.uint32
    # Comments show how a = 0x0000ffff is transformed.
    a = (a & 0x000000ff) | ((a & 0x0000ff00) << 8) # 0x00ff00ff
    a = (a & 0x000f000f) | ((a & 0x00f000f0) << 4) # 0x0f0f0f0f
    a = (a & 0x03030303) | ((a & 0x0c0c0c0c) << 2) # 0x33333333
    a = (a & 0x11111111) | ((a & 0x22222222) << 1) # 0x55555555
    return a

def roll_left(a, shift, length):
    """
    Rolls the bits in each element of a to the left by shift positions.
    """
    return ((a << shift) & ((1 << length) - 1)) | (a >> (length - shift))
