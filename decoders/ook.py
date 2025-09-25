import numpy as np
from dataclasses import dataclass
from typing import Tuple
from core.signal import Signal

def field_to_power(sig: Signal) -> Signal:
    """Photodiode view: P = |E|^2. Returns a real-valued Signal."""
    P = np.abs(sig.x)**2
    return Signal(x=P.astype(np.float64), fs=sig.fs, unit="W(a.u.)", meta={**sig.meta})

def symbols_from_power(power_sig: Signal, sps: int, offset: int = 0, reduce: str = "mean") -> np.ndarray:
    """
    Collapse oversampled power to one sample per symbol.
    reduce='mean' averages across each symbol; 'center' takes the center sample.
    """
    x = power_sig.x
    if reduce == "center":
        idx = np.arange(offset + sps//2, len(x), sps)
        return x[idx]
    # 'mean' (default)
    # Trim to multiple of sps starting at offset
    start = offset
    useful = x[start: start + ( (len(x)-start) // sps ) * sps ]
    if useful.size == 0:
        return np.array([], dtype=float)
    sym = useful.reshape(-1, sps).mean(axis=1)
    return sym

def threshold_1d_kmeans(v: np.ndarray, iters: int = 8) -> Tuple[float, float, float]:
    """
    Tiny 1-D k-means (K=2) to separate 'off' vs 'on'.
    Returns (mu0, mu1, thr) with mu0 < mu1 and thr = (mu0+mu1)/2.
    """
    if v.size == 0:
        return 0.0, 1.0, 0.5
    # init centers at p10 and p90
    c0, c1 = np.percentile(v, 10), np.percentile(v, 90)
    for _ in range(iters):
        d0 = np.abs(v - c0); d1 = np.abs(v - c1)
        g0 = v[d0 <= d1]; g1 = v[d1 < d0]
        # avoid empty clusters
        if g0.size: c0 = g0.mean()
        if g1.size: c1 = g1.mean()
    mu0, mu1 = (c0, c1) if c0 <= c1 else (c1, c0)
    thr = 0.5 * (mu0 + mu1)
    return mu0, mu1, thr

def slice_ook(sym_vals: np.ndarray, thr: float) -> np.ndarray:
    """Hard decisions: >= thr -> 1 else 0."""
    return (sym_vals >= thr).astype(np.uint8)

def bits_to_text(bits: np.ndarray) -> str:
    """
    Pack bits (MSB first per byte) back to a UTF-8 string.
    If length not multiple of 8, pad zeros at the end.
    """
    if bits.size % 8:
        pad = 8 - (bits.size % 8)
        bits = np.pad(bits, (0, pad), constant_values=0)
    byte_arr = np.packbits(bits)
    try:
        return byte_arr.tobytes().decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        # If your demo payload isn't valid UTF-8, fall back to 'replace' for visibility.
        return byte_arr.tobytes().decode("utf-8", errors="replace")

@dataclass
class OOKDecoder:
    sps: int
    offset: int = 0           # if you know symbol alignment; 0 for rectangular TX
    reduce: str = "mean"      # 'mean' or 'center'

    def decode(self, rx_field: Signal):
        is_elec = (rx_field.meta.get("domain") == "electrical")
        if is_elec:
            # already electrical after PD; don't square again
            x = rx_field.x.real
            P = Signal(x=x.astype(np.float64), fs=rx_field.fs, unit="V(a.u.)", meta={**rx_field.meta})
        else:
            P = field_to_power(rx_field)  # original path

        sym_vals = symbols_from_power(P, self.sps, self.offset, self.reduce)
        mu0, mu1, thr = threshold_1d_kmeans(sym_vals)
        bits = slice_ook(sym_vals, thr)
        info = {"mu0": float(mu0), "mu1": float(mu1), "thr": float(thr), "num_symbols": int(sym_vals.size)}
        return bits, "", info
