import numpy as np
from dataclasses import dataclass
from typing import Tuple
from core.signal import Signal
from phy.ethernet66 import align_66b_and_descramble, bits_to_bytes


def total_group_delay_samples(h_tx: np.ndarray) -> int:
    gd_tx = (len(h_tx) - 1) // 2
    gd_rx = gd_tx
    return gd_tx + gd_rx

def symbol_centers_indices(n_syms: int, sps: int, gd_total: int, N: int, extra_off: int = 0) -> np.ndarray:
    # NOTE: extra_off ∈ [0..sps-1] lets us test different symbol phases
    idx = gd_total + extra_off + np.arange(n_syms) * sps
    return idx[idx < N]

def viterbi_viterbi_qpsk_phase(z: np.ndarray) -> float:
    acc = np.sum(z**4)
    return 0.25 * np.angle(acc + 1e-30)

def matched_filter_rrc(rx_field: Signal, h_tx: np.ndarray) -> Signal:
    # RX: full + trim to input length (aligns to sample grid)
    h_rx = np.conj(h_tx[::-1])
    gd = (len(h_rx) - 1) // 2
    y_full = np.convolve(rx_field.x, h_rx, mode="full")
    y = y_full[gd : gd + len(rx_field.x)]
    return Signal(x=y, fs=rx_field.fs, unit=rx_field.unit,
                  meta={**rx_field.meta, "h_rx": h_rx, "gd_rx": gd})

def _circshift_bits(b: np.ndarray, k: int) -> np.ndarray:
    k %= b.size
    if k == 0: return b
    return np.concatenate([b[-k:], b[:-k]])

def qpsk_gray_slicer(z: np.ndarray) -> np.ndarray:
    # MUST be the inverse of bits_to_qpsk_gray (your current version is fine)
    Ipos = (np.real(z) >= 0)
    Qpos = (np.imag(z) >= 0)
    b0 = np.where(Ipos & Qpos, 0,
         np.where(~Ipos & Qpos, 0,
         np.where(~Ipos & ~Qpos, 1, 1))).astype(np.uint8)
    b1 = np.where(Ipos & Qpos, 0,
         np.where(~Ipos & Qpos, 1,
         np.where(~Ipos & ~Qpos, 1, 0))).astype(np.uint8)
    bits = np.empty(2*len(z), dtype=np.uint8)
    bits[0::2] = b0; bits[1::2] = b1
    return bits

def _swap_bit_pairs(bits: np.ndarray) -> np.ndarray:
    # true interleaved swap, not concatenation
    out = np.empty_like(bits)
    out[0::2] = bits[1::2]
    out[1::2] = bits[0::2]
    return out

def circshift_bits(b, k):
    k %= b.size
    if k == 0: return b
    return np.concatenate([b[-k:], b[:-k]])

@dataclass
class QPSKIdealDecoder:
    sps: int
    h_tx: np.ndarray
    do_phase_est: bool = True

    def decode(self, rx_field: Signal):
        y_mf = matched_filter_rrc(rx_field, self.h_tx)

        sps = self.sps
        n_syms_payload = int(len(rx_field.meta["syms"]))
        pad = int(rx_field.meta.get("pad_syms", 0))
        n_syms_total = int(rx_field.meta.get("n_syms_total", n_syms_payload + 2*pad))

        best = None
        angles = [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]

        for off in range(sps):
            # centers for the FULL (padded) stream
            idx_total = off + np.arange(n_syms_total) * sps
            idx_total = idx_total[idx_total < len(y_mf.x)]

            # payload-only centers (drop the guard symbols)
            start = pad
            stop  = pad + n_syms_payload
            idx = idx_total[start:stop]
            if idx.size == 0:
                continue

            z0 = y_mf.x[idx]
            phi_ff = viterbi_viterbi_qpsk_phase(z0) if (self.do_phase_est and len(z0) > 0) else 0.0
            phi_ff=0

            for a in angles:
                z = z0 * np.exp(-1j * (phi_ff + a))
                bits = qpsk_gray_slicer(z)

                # search remaining discrete ambiguities
                for order in ("normal", "swapped"):
                    raw = bits if order == "normal" else _swap_bit_pairs(bits)
                    for slip in (0, 1):
                        sraw = _circshift_bits(raw, slip)
                        for inv in (0, 1):
                            cand = sraw if inv == 0 else (sraw ^ 1)  # optional inversion
                            off66, _, mask = align_66b_and_descramble(cand)
                            score = int(mask.sum()) if mask is not None else 0
                            if (best is None) or (score > best["score"]):
                                best = {
                                    "score": score, "off": off, "phi_ff": float(phi_ff), "rot": float(a),
                                    "raw_bits": cand, "idx": idx, "off66": off66, "mask": mask,
                                    "order": order, "bit_slip": slip, "invert": inv
                                }

        if best is None:
            return np.array([], dtype=np.uint8), "", {"num_symbols": 0, "note": "no valid hypothesis"}

        # Decode using the best candidate (and optionally keep only good blocks)
        off66, payload_bits, mask = align_66b_and_descramble(best["raw_bits"])
        payload_bytes = bits_to_bytes(payload_bits)
        try:
            text = payload_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            text = payload_bytes.decode("utf-8", errors="replace")

        info = {
            "num_symbols": int(len(best["raw_bits"]) // 2),
            "phase_ff_rad": best["phi_ff"],
            "extra_symbol_offset": int(best["off"]),
            "quadrant_rot_rad": best["rot"],
            "pair_order": best["order"],
            "bit_slip": int(best["bit_slip"]),
            "invert": int(best["invert"]),
            "align66_offset_bits": int(best["off66"]),
            "valid_headers": int(best["mask"].sum()) if best["mask"] is not None else 0,
            "score": int(best["score"]),
            "first_sample_indices": best["idx"][:10].tolist()
        }
        return payload_bits, text, info

