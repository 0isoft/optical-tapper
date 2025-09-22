import numpy as np
from typing import Tuple

# 64b/66b essentials we’ll model:
# - Each 66b block = 2-bit sync header + 64 data bits
# - Data block header = '01' (control blocks use '10'; we’ll use '01' here)
# - Self-synchronous scrambler: s[k] = b[k] XOR s[k-39] XOR s[k-58]
#   (IEEE 802.3 64b/66b PCS scrambler polynomial x^58 + x^39 + 1)

SYNC_DATA = np.array([0,1], dtype=np.uint8)   # '01'

def bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size % 8:
        bits = np.pad(bits, (0, 8 - bits.size % 8))
    return np.packbits(bits).tobytes()

def chunk64_pad(bits: np.ndarray) -> np.ndarray:
    """Group payload bits into 64-bit blocks (pad zeros at the end if needed)."""
    pad = (-len(bits)) % 64
    if pad:
        bits = np.pad(bits, (0, pad), constant_values=0)
    return bits.reshape(-1, 64)

def scramble_ss(bits: np.ndarray) -> np.ndarray:
    """
    Self-synchronous scrambler (bitwise). Input: raw payload bits (concatenated).
    Output: scrambled bits s[k] = b[k] XOR s[k-39] XOR s[k-58]; s[k<0]=0.
    """
    s = np.zeros_like(bits, dtype=np.uint8)
    for k in range(bits.size):
        t39 = s[k-39] if k >= 39 else 0
        t58 = s[k-58] if k >= 58 else 0
        s[k] = bits[k] ^ t39 ^ t58
    return s

def descramble_ss(s: np.ndarray) -> np.ndarray:
    """
    Inverse of self-synchronous scrambler (same recurrence, using past *scrambled* bits).
    b[k] = s[k] XOR s[k-39] XOR s[k-58].
    """
    b = np.zeros_like(s, dtype=np.uint8)
    for k in range(s.size):
        t39 = s[k-39] if k >= 39 else 0
        t58 = s[k-58] if k >= 58 else 0
        b[k] = s[k] ^ t39 ^ t58
    return b

def encode_64b66b(payload_bytes: bytes) -> np.ndarray:
    """
    Produce a realistic 66b bitstream: [01][64 scrambled bits] repeated.
    (We only model data blocks; control blocks '10' omitted for simplicity.)
    """
    bits = bytes_to_bits(payload_bytes)
    blocks = chunk64_pad(bits)                  # shape: (N, 64)
    flat64 = blocks.reshape(-1)
    scr = scramble_ss(flat64)                   # concatenate, then scramble
    # Re-split into 64-bit payloads and prefix '01' to each
    N = blocks.shape[0]
    scr_blocks = scr.reshape(N, 64)
    out = []
    for k in range(N):
        out.append(SYNC_DATA)
        out.append(scr_blocks[k])
    return np.concatenate(out)

def align_66b_and_descramble(rx_bits: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Find 66-bit block boundary by maximizing the fraction of headers == '01'.
    Returns (offset, payload_bits (descrambled, concat), header_mask)
    header_mask[k]=1 where a valid '01' header was found.
    """
    if rx_bits.size < 66:
        return 0, np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)

    best_off, best_score = 0, -1
    best_mask = None
    for off in range(66):
        # How many complete 66b blocks starting at this offset?
        count = (rx_bits.size - off) // 66
        if count <= 0: 
            continue
        hdrs = rx_bits[off : off + 66*count].reshape(count, 66)[:, :2]
        mask = np.all(hdrs == SYNC_DATA, axis=1).astype(np.uint8)
        score = mask.sum()
        if score > best_score:
            best_score = score
            best_off = off
            best_mask = mask

    # Extract payload bits from aligned blocks, keep only blocks with good headers
    count = (rx_bits.size - best_off) // 66
    blk = rx_bits[best_off : best_off + 66*count].reshape(count, 66)
    good = best_mask.astype(bool)
    payload_scr = blk[good, 2:]                # (G, 64)
    if payload_scr.size == 0:
        return best_off, np.array([], dtype=np.uint8), best_mask
    payload_scr_flat = payload_scr.reshape(-1)
    payload_bits = descramble_ss(payload_scr_flat)
    return best_off, payload_bits, best_mask
