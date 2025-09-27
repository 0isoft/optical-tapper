import numpy as np
import matplotlib.pyplot as plt

def nrz_from_bits(bits, sps, hi=1.0, lo=0.0):
    """Rectangular NRZ from bit array."""
    levels = np.where(bits > 0, hi, lo).astype(float)
    return np.repeat(levels, sps)

def eye_plot(x, sps, span_sym=2, ntraces=200, title="Eye", yscale=1.0, ylabel="Amplitude"):
    seglen = span_sym * sps
    n = min(ntraces, (len(x) - seglen) // sps)
    if n <= 0: 
        return
    import matplotlib.pyplot as plt
    plt.figure()
    for k in range(n):
        i0 = k * sps
        seg = x[i0:i0+seglen] * yscale
        t = (np.arange(seglen) - seglen/2) / sps
        plt.plot(t, seg, linewidth=0.6)
    plt.grid(True, alpha=0.3); plt.title(title)
    plt.xlabel("UI"); plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=200)
    plt.close()

def auto_best_offset(rx_elec, sps, slicer):
    """Try offsets 0..sps-1 and pick the one maximizing header hits."""
    from phy.ethernet66 import align_66b_and_descramble
    best = None
    for off in range(sps):
        sym_vals = _symvals(rx_elec, sps, off, reduce="center")
        thr = _kmeans_thr(sym_vals)
        raw_bits = (sym_vals >= thr).astype(np.uint8)
        _, _, mask = align_66b_and_descramble(raw_bits)
        score = int(mask.sum())
        if (best is None) or (score > best["score"]):
            best = {"off": off, "score": score, "thr": float(thr)}
    return best

def _symvals(x, sps, off=0, reduce="center"):
    if reduce == "center":
        idx = np.arange(off + sps//2, len(x), sps)
        return x[idx]
    start = off
    useful = x[start:start + ((len(x)-start)//sps)*sps]
    if useful.size == 0: return np.array([], float)
    return useful.reshape(-1, sps).mean(axis=1)

def _kmeans_thr(v, iters=8):
    if v.size == 0: return 0.5*(v.min()+v.max()) if v.size else 0.5
    c0, c1 = np.percentile(v, 10), np.percentile(v, 90)
    for _ in range(iters):
        d0, d1 = np.abs(v-c0), np.abs(v-c1)
        g0, g1 = v[d0<=d1], v[d1<d0]
        if g0.size: c0 = g0.mean()
        if g1.size: c1 = g1.mean()
    mu0, mu1 = (c0, c1) if c0 <= c1 else (c1, c0)
    return 0.5*(mu0+mu1)

def xcorr_delay(a, b):
    """Return sample lag (b delayed vs a) maximizing cross-correlation."""
    a = (a - a.mean()); b = (b - b.mean())
    n = min(len(a), len(b))
    r = np.correlate(b[:n], a[:n], mode="full")
    lag = np.argmax(r) - (n-1)
    return lag

def rise_fall_10_90(x, sps, search=4):
    """
    Estimate 10-90% rise and 90-10% fall times (in samples) by scanning for edges.
    search: number of symbols to scan for edges.
    """
    # normalize 0..1 by percentiles for robustness
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    xn = (x - lo) / max(hi - lo, 1e-12)
    # find edges near first 'search' symbol periods
    win = search * sps
    xr = xn[:win]
    # rising edge: first sample where derivative positive and mean below 0.5
    d = np.diff(xr)
    if d.size < 2: return None, None
    idx = np.argmax(d)  # coarse
    seg = xr[max(idx-2*sps,0):idx+2*sps]
    t = np.arange(len(seg))
    t10 = _first_cross(seg, 0.1); t90 = _first_cross(seg, 0.9)
    tr = (t90 - t10) if (t10 is not None and t90 is not None) else None
    # falling edge similarly (invert)
    segf = 1.0 - seg
    f10 = _first_cross(segf, 0.1); f90 = _first_cross(segf, 0.9)
    tf = (f90 - f10) if (f10 is not None and f90 is not None) else None
    return tr, tf

def _first_cross(x, y):
    above = x >= y
    idx = np.where(np.diff(above.astype(int))>0)[0]
    return int(idx[0]) if idx.size else None

def plot_waveforms(t, traces, labels, title):
    plt.figure()
    for y, lb in zip(traces, labels):
        plt.plot(t, y, label=lb, linewidth=0.9)
    plt.grid(True, alpha=0.3); plt.legend()
    plt.title(title); plt.xlabel("time [s]"); plt.ylabel("amplitude")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=200)
    plt.close()
