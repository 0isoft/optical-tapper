# utils/metrics.py
import numpy as np
import matplotlib.pyplot as plt
from phy.ethernet66 import align_66b_and_descramble

def symbol_centers(x: np.ndarray, sps: int, off: int) -> np.ndarray:
    idx = off + np.arange(len(x)//sps) * sps
    idx = idx[idx < len(x)]
    return x[idx]

def eye_stats(sym: np.ndarray):
    # quick 2-cluster threshold from percentiles
    p15, p85 = np.percentile(sym, 15), np.percentile(sym, 85)
    thr = 0.5*(p15 + p85)
    g0 = sym[sym <  thr]
    g1 = sym[sym >= thr]
    mu0 = float(np.mean(g0)) if g0.size else 0.0
    mu1 = float(np.mean(g1)) if g1.size else 1.0
    s0  = float(np.std(g0, ddof=1)) if g0.size > 1 else 1e-12
    s1  = float(np.std(g1, ddof=1)) if g1.size > 1 else 1e-12
    # one-number SNR (per-level variances averaged)
    snr_linear = ((mu1 - mu0)**2) / (s0**2 + s1**2 + 1e-30)
    snr_db = 10*np.log10(snr_linear + 1e-30)
    return {"thr":thr, "mu0":mu0, "mu1":mu1, "s0":s0, "s1":s1, "snr_db":snr_db}

def empirical_ber(bits_rx: np.ndarray, bits66_tx: np.ndarray):
    """
    Compare *payload bits after descramble* against ground truth.
    Ground truth is obtained by running align_66b_and_descramble on the original 66b stream.
    """
    off_tx, payload_tx, mask_tx = align_66b_and_descramble(bits66_tx.astype(np.uint8))
    off_rx, payload_rx, mask_rx = align_66b_and_descramble(bits_rx.astype(np.uint8))

    # Trim to common length
    L = min(len(payload_tx), len(payload_rx))
    if L == 0:
        return {"ber":1.0, "nerr":0, "nbits":0, "valid_tx":int(mask_tx.sum()), "valid_rx":int(mask_rx.sum())}
    diff = payload_tx[:L] ^ payload_rx[:L]
    nerr = int(diff.sum())
    ber  = nerr / L
    return {
        "ber": ber, "nerr": nerr, "nbits": L,
        "valid_tx": int(mask_tx.sum()), "valid_rx": int(mask_rx.sum()),
        "off_tx": int(off_tx), "off_rx": int(off_rx)
    }

def plot_ber_points(points, title="BER comparison (tap vs normal)"):
    """
    points = [("tap", ber_tap), ("normal", ber_norm)]
    """
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

def plot_snr_bars(pairs, title="SNR at symbol centers"):
    """
    pairs = [("tap", snr_db_tap), ("normal", snr_db_norm)]
    """
    names = [p[0] for p in pairs]
    snrs  = [p[1] for p in pairs]
    xs = np.arange(len(snrs))
    plt.figure()
    plt.bar(xs, snrs)
    plt.xticks(xs, names)
    plt.grid(True, axis="y", alpha=0.3)
    plt.ylabel("SNR [dB]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("snr_bars.png", dpi=200)
    plt.close()
