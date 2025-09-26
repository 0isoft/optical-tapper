# utils/sanity.py
import numpy as np
from math import erfc, sqrt

def seed_everything(seed=42):
    np.random.seed(seed)

def assert_shapes(*arrs):
    names = [f"a{i}" for i,_ in enumerate(arrs)]
    Ls = [len(a) for a in arrs]
    if len(set(Ls)) != 1:
        raise ValueError(f"[sanity] length mismatch: " + ", ".join(f"{n}={L}" for n,L in zip(names,Ls)))

def check_fs_sps(fs, Rs, sps):
    exp = Rs * sps
    if abs(fs - exp)/exp > 1e-12:
        raise ValueError(f"[sanity] fs mismatch: fs={fs:g} vs Rs*sps={exp:g}")

def check_bandwidths(tx_bw_hz, rx_bw_hz, fs, label=""):
    nyq = fs/2
    warn = []
    if tx_bw_hz > nyq*0.95: warn.append(f"TX BW {tx_bw_hz/1e9:.2f} GHz ~ Nyquist {nyq/1e9:.2f} GHz")
    if rx_bw_hz > nyq*0.95: warn.append(f"RX BW {rx_bw_hz/1e9:.2f} GHz ~ Nyquist {nyq/1e9:.2f} GHz")
    if warn: print("[sanity]", label, " | ".join(warn))

def check_power_scaling(Ptx_avg_W, loss_dB, P_unit_avg, kW):
    Prx_avg_W = Ptx_avg_W * (10**(-loss_dB/10))
    if not (0.25*Prx_avg_W <= P_unit_avg*kW <= 4.0*Prx_avg_W):
        print(f"[sanity] power scale off: expected≈{Prx_avg_W:.3e}W, got≈{P_unit_avg*kW:.3e}W (kW={kW:.3e})")

def check_coupling(coup_dB):
    if coup_dB > -10:
        print(f"[sanity] optical coupling {coup_dB} dB is *very* strong for a passive tap.")
    if coup_dB < -40:
        print(f"[sanity] optical coupling {coup_dB} dB is extremely weak; SNR may collapse.")

def adc_headroom(v, vref, tag):
    vpk = np.max(np.abs(v))
    util = vpk / (vref + 1e-30)
    if util > 1.0:
        print(f"[sanity][{tag}] ADC clipping: peak={vpk:.3g} > vref={vref:.3g} (util={util:.2f})")
    elif util < 0.05:
        print(f"[sanity][{tag}] ADC under-utilized: peak={vpk:.3g} << vref={vref:.3g} (util={util:.2f})")
    return util

def symbol_centers_trace(x, sps, off):
    idx = off + np.arange(max(0, (len(x)-off)//sps)) * sps + sps//2
    idx = idx[idx < len(x)]
    return x[idx]

def eye_from_symbols(sym):
    if sym.size == 0:
        return dict(mu0=0, mu1=1, s0=1e-9, s1=1e-9, thr=0.5)
    p10, p90 = np.percentile(sym, 10), np.percentile(sym, 90)
    c0, c1 = p10, p90
    for _ in range(8):
        d0 = np.abs(sym-c0); d1 = np.abs(sym-c1)
        g0 = sym[d0 <= d1]; g1 = sym[d1 < d0]
        if g0.size: c0 = g0.mean()
        if g1.size: c1 = g1.mean()
    mu0, mu1 = (c0,c1) if c0<=c1 else (c1,c0)
    thr = 0.5*(mu0+mu1)
    z0 = sym[sym <  thr]; z1 = sym[sym >= thr]
    s0 = z0.std(ddof=1) if z0.size>1 else 1e-9
    s1 = z1.std(ddof=1) if z1.size>1 else 1e-9
    return dict(mu0=float(mu0), mu1=float(mu1), s0=float(s0), s1=float(s1), thr=float(thr))

def q_and_ber(mu0, mu1, s0, s1):
    Q = (mu1 - mu0) / (s0 + s1 + 1e-30)
    ber = 0.5 * erfc(Q / sqrt(2))
    return Q, ber

import numpy as np

def _crop_blocks(payload, mask, Nkeep, start=0, k_payload_bits=64):
    # payload: flat bits, mask: per-block
    payload = np.asarray(payload, dtype=np.uint8)
    mask    = np.asarray(mask,    dtype=bool)

    Nblocks = min(len(mask), payload.size // k_payload_bits)
    if Nblocks <= 0:
        return np.empty(0, np.uint8), np.empty(0, bool)

    payload = payload[:Nblocks * k_payload_bits].reshape(Nblocks, k_payload_bits)
    # window selection with safety bounds
    s = max(0, min(Nblocks, start))
    e = max(0, min(Nblocks, start + Nkeep))
    if e <= s:
        return np.empty(0, np.uint8), np.empty(0, bool)

    return payload[s:e].reshape(-1), mask[s:e]


def masked_payload(tx_payload, tx_mask, rx_payload, rx_mask, k_payload_bits=64, shift_search=16):
    """
    Align TX/RX 64b payloads using per-block masks AND an integer block shift search.
    We pick the shift (Δ blocks) that minimizes BER over the overlap.
    Returns flattened aligned payloads (same length). If no overlap, returns length-0 arrays.
    """
    tx_payload = np.asarray(tx_payload, dtype=np.uint8)
    rx_payload = np.asarray(rx_payload, dtype=np.uint8)
    tx_mask    = np.asarray(tx_mask,    dtype=bool)
    rx_mask    = np.asarray(rx_mask,    dtype=bool)

    # Derive block counts from payload lengths (more reliable than masks alone)
    Ntx = min(tx_mask.size, tx_payload.size // k_payload_bits)
    Nrx = min(rx_mask.size, rx_payload.size // k_payload_bits)
    if Ntx == 0 or Nrx == 0:
        return np.empty(0, np.uint8), np.empty(0, np.uint8)

    # Pre-reshape to [Nblocks, 64] for cheap slicing
    tx_mat = tx_payload[:Ntx * k_payload_bits].reshape(Ntx, k_payload_bits)
    rx_mat = rx_payload[:Nrx * k_payload_bits].reshape(Nrx, k_payload_bits)
    tx_mask = tx_mask[:Ntx]
    rx_mask = rx_mask[:Nrx]

    best = None
    # search Δ in [-shift_search, +shift_search]
    for d in range(-shift_search, shift_search + 1):
        # TX window: [t0, t1), RX window shifted by d blocks: [r0, r1)
        if d >= 0:
            t0, r0 = 0, d
            N = min(Ntx, Nrx - d)
        else:
            t0, r0 = -d, 0
            N = min(Ntx + d, Nrx)  # d negative

        if N <= 0:
            continue

        # masks over overlap
        mt = tx_mask[t0:t0+N]
        mr = rx_mask[r0:r0+N]
        m  = mt & mr
        if not np.any(m):
            continue

        # payload rows over overlap
        trows = tx_mat[t0:t0+N][m]
        rrows = rx_mat[r0:r0+N][m]
        if trows.size == 0:
            continue

        # quick BER on overlapping valid rows
        e = int(np.count_nonzero((trows ^ rrows).astype(np.uint8)))
        nb = int(trows.size * k_payload_bits)
        # (rows already k_payload_bits wide; flatten size = rows*k)
        # but since we didn't flatten, treat per-bit directly:
        nb = int(trows.size * trows.shape[1])  # same as rows*64
        ber = e / nb if nb else 1.0

        if (best is None) or (ber < best["ber"]):
            best = {"d": d, "ber": ber, "rows": trows.copy(), "rows_rx": rrows.copy()}

    if best is None:
        return np.empty(0, np.uint8), np.empty(0, np.uint8)

    # Flatten selected rows
    tx_sel = best["rows"].reshape(-1).astype(np.uint8)
    rx_sel = best["rows_rx"].reshape(-1).astype(np.uint8)
    return tx_sel, rx_sel


def masked_payload_ber(tx_payload, tx_mask, rx_payload, rx_mask, k_payload_bits=64, shift_search=16):
    tx_sel, rx_sel = masked_payload(
        tx_payload, tx_mask, rx_payload, rx_mask,
        k_payload_bits=k_payload_bits, shift_search=shift_search
    )
    N = int(tx_sel.size)
    if N == 0:
        return dict(ber=1.0, nerr=0, nbits=0, ok=False)
    e = int(np.count_nonzero((tx_sel ^ rx_sel).astype(np.uint8)))
    return dict(ber=e / N, nerr=e, nbits=N, ok=True)



def check_ac_hpf(ac_hz, Rs, tag):
    # heuristic: HPF << line rate/100 keeps wander minimal for random payloads
    if ac_hz > Rs/100:
        print(f"[sanity][{tag}] AC-HPF ({ac_hz:.2g} Hz) is fast relative to Rs={Rs:.3g} — expect baseline wander.")

def warn_low_valid66(valid, total_blocks=None, tag=""):
    if valid == 0:
        print(f"[sanity][{tag}] 66b alignment found 0 valid headers.")
    elif total_blocks and valid < 0.5*total_blocks:
        print(f"[sanity][{tag}] only {valid}/{total_blocks} 66b headers valid (~{100*valid/total_blocks:.1f}%).")
