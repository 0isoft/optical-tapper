# utils/metrics.py
import numpy as np
import matplotlib.pyplot as plt
from math import erfc, sqrt
from utils.ook_vis import _kmeans_thr, xcorr_delay  # use the same thresholding as the decoder

def symbol_centers(x: np.ndarray, sps: int, off: int) -> np.ndarray:
    idx = off + np.arange(len(x)//sps) * sps
    idx = idx[idx < len(x)]
    return x[idx]

def align_symbols_and_bits(symvals: np.ndarray, rx_bits_sym: np.ndarray, tx_bits: np.ndarray, nskip_syms: int = 2000):
    """
    Align symbol-center samples 'symvals' to the true TX bits by
    estimating the integer bit lag between the RX symbol decisions and TX bits.
    Returns sym_aln, tx_aln after applying the lag and skipping warm-up.
    """
    v = np.asarray(symvals, float).ravel()
    rb = np.asarray(rx_bits_sym, np.uint8).ravel()
    tx = np.asarray(tx_bits,     np.uint8).ravel()

    if v.size == 0 or rb.size == 0 or tx.size == 0:
        return np.array([], float), np.array([], np.uint8)

    # Estimate bit lag: "rb delayed vs tx by lag" (per xcorr_delay docstring)
    lag = int(xcorr_delay(tx.astype(float), rb.astype(float)))

    # Align rb and tx by trimming lead/trail; then trim symvals to the same
    if lag >= 0:
        # rb occurs later ⇒ drop 'lag' bits from TX head
        tx_aln = tx[lag:]
        rb_aln = rb[:len(tx_aln)]
    else:
        # rb occurs earlier ⇒ drop '-lag' from rb head
        rb_aln = rb[-lag:]
        tx_aln = tx[:len(rb_aln)]

    L = min(len(v), len(rb_aln), len(tx_aln))
    if L <= nskip_syms:
        return np.array([], float), np.array([], np.uint8)

    # Apply warm-up skip to avoid DC-restorer/transient garbage
    v_aln  = v[:L][nskip_syms:]
    tx_aln = tx_aln[:L][nskip_syms:]

    return v_aln, tx_aln

def eye_stats_da(symvals: np.ndarray, tx_bits: np.ndarray):
    """
    Data-aided eye stats using already aligned symbol samples and TX bits.
    """
    v = np.asarray(symvals, float).ravel()
    b = np.asarray(tx_bits,  np.uint8).ravel()
    L = min(v.size, b.size)
    if L < 32:
        return {"mu0":0.0,"mu1":1.0,"s0":1e-12,"s1":1e-12,"Q":0.0,"ber_th":0.5}
    v = v[:L]; b = b[:L]
    v0 = v[b==0]; v1 = v[b==1]
    if v0.size < 2 or v1.size < 2:
        return {"mu0":0.0,"mu1":1.0,"s0":1e-12,"s1":1e-12,"Q":0.0,"ber_th":0.5}
    mu0, mu1 = float(v0.mean()), float(v1.mean())
    s0,  s1  = float(v0.std(ddof=1)), float(v1.std(ddof=1))
    Q = (mu1 - mu0) / (s0 + s1 + 1e-30)
    ber_th = 0.5 * erfc(Q / sqrt(2.0))
    return {"mu0":mu0,"mu1":mu1,"s0":s0,"s1":s1,"Q":Q,"ber_th":ber_th}

def empirical_ber_aligned(rx_bits_aligned: np.ndarray, tx_bits_aligned: np.ndarray):
    rx = np.asarray(rx_bits_aligned, np.uint8).ravel()
    tx = np.asarray(tx_bits_aligned, np.uint8).ravel()
    L = min(rx.size, tx.size)
    if L == 0: return {"ber":1.0, "nerr":0, "nbits":0}
    diff = (rx[:L] ^ tx[:L]).astype(np.uint8)
    nerr = int(diff.sum())
    return {"ber": nerr/L, "nerr": nerr, "nbits": L}

def plot_ber_points(points, title="BER comparison (tap vs normal)"):
    names = [p[0] for p in points]
    bers  = [max(p[1], 1e-12) for p in points]  # avoid log(0)
    xs = np.arange(len(bers))
    plt.figure()
    plt.semilogy(xs, bers, "o", markersize=7)
    plt.xticks(xs, names)
    plt.grid(True, which="both", alpha=0.3)
    plt.ylabel("BER (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("ber_points.png", dpi=200)
    plt.close()

def plot_snr_bars(pairs, title="Eye SNR / Q (symbol centers)"):
    names = [p[0] for p in pairs]
    snrs  = [p[1] for p in pairs]
    xs = np.arange(len(snrs))
    plt.figure()
    plt.bar(xs, snrs)
    plt.xticks(xs, names)
    plt.grid(True, axis="y", alpha=0.3)
    plt.ylabel("Eye SNR [dB]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("snr_bars.png", dpi=200)
    plt.close()

def payload_symbol_indices(mask_blocks: np.ndarray, off66: int, total_syms: int) -> np.ndarray:
    """
    Return the flat symbol indices (0..total_syms-1) that belong to the *payload*
    bits of every 66b block where mask_blocks[k] == True, given the 66b block offset 'off66'.
    Header bits are positions [0,1] inside each 66b block; payload is [2..65].

    mask_blocks : shape (Nblocks,)
    off66       : integer start offset of the first 66b block in the bitstream
    total_syms  : total number of symbol-center samples available
    """
    mask_blocks = np.asarray(mask_blocks, bool).ravel()
    if mask_blocks.size == 0:
        return np.array([], dtype=int)

    # For block k, block start symbol index is: start = off66 + 66*k
    k_idx = np.nonzero(mask_blocks)[0]
    starts = off66 + 66 * k_idx[:, None]  # shape (K,1)

    # Payload positions 2..65 (inclusive)
    payload_pos = np.arange(2, 66, dtype=int)[None, :]  # shape (1,64)

    idx = (starts + payload_pos).reshape(-1)  # flattened indices
    idx = idx[(idx >= 0) & (idx < total_syms)]  # guard bounds
    return idx.astype(int)

def eye_stats_da_from_indices(symvals_all: np.ndarray, tx_bits66: np.ndarray, payload_idx: np.ndarray):
    """
    Data-aided eye stats using precomputed payload symbol indices.
    symvals_all : symbol-center samples (same indexing as bits66)
    tx_bits66   : ground-truth 66b bitstream (0/1) used to label symbols
    payload_idx : indices into symvals_all/bits66 selecting valid payload symbols
    """
    if payload_idx.size < 64:  # too few points for a robust estimate
        return {"mu0":0.0,"mu1":1.0,"s0":1e-12,"s1":1e-12,"Q":0.0,"ber_th":0.5}

    v = np.asarray(symvals_all, float).ravel()
    b = np.asarray(tx_bits66,  np.uint8).ravel()
    L = min(v.size, b.size)
    payload_idx = payload_idx[payload_idx < L]

    if payload_idx.size < 64:
        return {"mu0":0.0,"mu1":1.0,"s0":1e-12,"s1":1e-12,"Q":0.0,"ber_th":0.5}

    vv = v[payload_idx]
    bb = b[payload_idx]

    v0 = vv[bb == 0]
    v1 = vv[bb == 1]
    if v0.size < 2 or v1.size < 2:
        return {"mu0":0.0,"mu1":1.0,"s0":1e-12,"s1":1e-12,"Q":0.0,"ber_th":0.5}

    mu0, mu1 = float(v0.mean()), float(v1.mean())
    s0,  s1  = float(v0.std(ddof=1)), float(v1.std(ddof=1))

    # Ensure "1" is the higher level for a positive Q
    if mu1 < mu0:
        mu0, mu1, s0, s1 = mu1, mu0, s1, s0

    Q = (mu1 - mu0) / (s0 + s1 + 1e-30)
    ber_th = 0.5 * erfc(Q / sqrt(2.0))

    # EVM-style SNR: average symbol spacing over noise power
    snr_lin = ((mu1 - mu0)**2) / (s0**2 + s1**2 + 1e-30)
    snr_db  = 10*np.log10(max(snr_lin, 1e-30))

    return {"mu0":mu0,"mu1":mu1,"s0":s0,"s1":s1,"Q":Q,"ber_th":ber_th,"snr_db":snr_db}